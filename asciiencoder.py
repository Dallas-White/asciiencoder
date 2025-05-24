'''
    ASCII encoder.
    Copyright (C) <2025>  <Dallas White>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
'''


import os.path
import subprocess
from PIL import Image, ImageDraw, ImageFont
import ffmpeg
from functools import partial
from multiprocessing import Pool
import math
import numpy
from tqdm import tqdm
import time
import string
import argparse
import cv2
import sys


def get_largest_bounding_box(font, characters):
    largest_width = 0
    largest_height = 0
    for x in characters:
        bbox = font.getbbox(x)
        if(bbox[2] > largest_width):
            largest_width = bbox[2]
        if(bbox[3] > largest_height):
            largest_height = bbox[3]
    return (0,0,largest_width, largest_height)


def calculate_font_dimensions(font, characters):
    bbox = get_largest_bounding_box(font, characters)
    return (bbox[2]-bbox[0],bbox[3]-bbox[1])


#calculate the percentage of white characterse in a cached image
def calculate_opacity_percent(img):
    total_pixels = img.size
    white_pixels = numpy.sum(img == 255)
    opacity_percent = (white_pixels / total_pixels) * 100
    return opacity_percent

# generate a character cache for a given font and characters
def generate_character_cache(font,characters):
    cached_images = []
    font_size= calculate_font_dimensions(font,characters)
    img_size = (font_size[0]*2,font_size[1]*2)
    bounding_box = get_largest_bounding_box(font, characters)
    for x in characters:
        img = Image.new("RGB",img_size, color=(0,0,0))
        draw = ImageDraw.Draw(img)
        draw.text((0,0), x, fill=(255,255,255), font=font)
        img = img.crop((bounding_box)).convert("L").convert("RGB")
        image_array = numpy.array(img)
        non_black_mask = numpy.any(image_array != 0, axis=2)
        character_mask = numpy.zeros(image_array.shape[:2], dtype=numpy.uint8)
        character_mask[non_black_mask] = 255
        cached_images.append(character_mask)
    

    sorted_images = sorted(cached_images, key=calculate_opacity_percent)
    image_map = {}
    for pixel in range(0,256):
        ascii_index = int((pixel * len(characters)) / 256)
        image_map[pixel] = sorted_images[ascii_index]
    return image_map

#determins the optimal font size for a font within the given range, returns (None,None,None) if none exists
def find_optimal_font_size(font_path, output_height, output_width, characters,min_font_size, max_font_size):
    for font_size in range(min_font_size, max_font_size+1):
        font = ImageFont.truetype(font_path, font_size)
        font_width, font_height = calculate_font_dimensions(font,characters)
        if output_width % font_width == 0 and output_height % font_height == 0:
            rows = output_height // font_height
            cols = output_width // font_width
            return (rows, cols,font_size)
    return (None,None,None)

#an Iterator to lazily load video frames
def video_frame_iterator(video, start=None, count=None):
    frame_count = -1
    while True:
        frame_count += 1
        ret, frame = video.read()
        if not ret:
           return 
        if start is not None and frame_count < start:
            continue
        yield frame
        if count is not None and frame_count > count+start:
            return


#converts a color pallete into a PIL compatable pallete
def create_palette_image(rgb_palette):
    flat_palette = [value for color in rgb_palette for value in color]
    
    while len(flat_palette) < 768:
        flat_palette.extend([0, 0, 0]) 

    flat_palette = flat_palette[:768]

    palette_img = Image.new("P", (1, 1))
    palette_img.putpalette(flat_palette)

    return palette_img

ansi_16_colors = [
    (0, 0, 0),       # black
    (128, 0, 0),     # red
    (0, 128, 0),     # green
    (128, 128, 0),   # yellow
    (0, 0, 128),     # blue
    (128, 0, 128),   # magenta
    (0, 128, 128),   # cyan
    (192, 192, 192), # light gray
    (128, 128, 128), # dark gray
    (255, 0, 0),     # bright red
    (0, 255, 0),     # bright green
    (255, 255, 0),   # bright yellow
    (0, 0, 255),     # bright blue
    (255, 0, 255),   # bright magenta
    (0, 255, 255),   # bright cyan
    (255, 255, 255), # white
]

ansi_16 = create_palette_image(ansi_16_colors)

def build_ansi_256():
    palette = ansi_16_colors.copy()
    steps = [0, 95, 135, 175, 215, 255]
    for r in steps:
        for g in steps:
            for b in steps:
                palette.append((r, g, b))

    for gray in range(8, 248, 10):
        palette.append((gray, gray, gray))

    return palette

ansi_256 = create_palette_image(build_ansi_256())

#the main function that converts a frame into an ascii frame
def transform_frame(frame,font, rows, cols, characters, font_width, font_height, args):
    img = Image.fromarray(frame)
    if args.color == "BW" or args.color == "grey" or args.color == "red" or args.color == "green" or args.color == "blue":
        img = img.convert("L").convert("RGB")
    elif args.color == "16":
        img = img.quantize(palette=ansi_16).convert("RGB")
    elif args.color == "256":
        img = img.quantize(palette=ansi_256).convert("RGB")
    
    if args.color == "red":
        r, g, b = img.split()
        img = Image.merge('RGB', (r, Image.new('L', img.size, 0), Image.new('L', img.size, 0)))
    elif args.color == "green":
        r, g, b = img.split()
        img = Image.merge('RGB', (Image.new('L', img.size, 0), g, Image.new('L', img.size, 0)))
    elif args.color == "blue":
        r, g, b = img.split()
        img = Image.merge('RGB', (Image.new('L', img.size, 0), Image.new('L', img.size, 0), b))

    img = img.resize((cols,rows))
    background_color = args.background_color
    output_img = numpy.full((rows*font_height, cols*font_width ,3),background_color,dtype=numpy.uint8)
    pixels = img.load()
    for col in range(cols):
        for row in range(rows):
            pixel = pixels[col,row]
            x = font_width * col 
            y = font_height * row 
            fg_color =  (255,255,255)
            grey_pixel = (pixel[0] + pixel[1] + pixel[2]) // 3
            bg_grey = (background_color[0] + background_color[1] + background_color[2]) // 3
            char_mask = characters[abs(grey_pixel-bg_grey)]
            h, w = char_mask.shape
            if args.color != "BW":
                fg_color = pixel
            output_img[y:y+font_height, x:x+font_width ,:][char_mask == 255] = fg_color

    return numpy.array(output_img)

# a parser for rgb colors in the format r,g,b
def rgb_color(s):
    try:
        parts = s.split(',')
        if len(parts) != 3:
            raise ValueError
        rgb = tuple(int(p) for p in parts)
        if any(c < 0 or c > 255 for c in rgb):
            raise ValueError
        return rgb
    except ValueError:
        raise argparse.ArgumentTypeError(f"Color must be 3 integers 0-255 separated by commas, got '{s}'")

# a parser for resolution
def parse_resolution(r):
    parts = r.split("x")
    if len(parts) != 2:
        raise ValueError
    try:
        return tuple(int(p) for p in parts)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Resolution must be in the format HxW Example: 720x1280")

if __name__ == "__main__":
    CHARACTERS = list(string.ascii_letters) + list(string.digits) + list(string.punctuation)
    parser = argparse.ArgumentParser(description="convert videos into ascii movies", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("video_path", help="Path to the video file.")
    parser.add_argument("font_path",help="Path to a ttf or otf font file. (Monospace fonts are recommended)")
    parser.add_argument("output_path", help="Path to save the ASCII movie.")
    parser.add_argument("--resolution", help="Resolution of the ASCII movie. (Default: 720x1280)", default="720x1280", type=parse_resolution)
    parser.add_argument("--min-font-size", help="Minimum font size in pixels. (Default: 8)", default=8, type=int)
    parser.add_argument("--max-font-size", help="Maximum font size in pixels. (Default: 24)", default=24, type=int)
    parser.add_argument("--color", choices=["BW","grey", "red", "green", "blue","16","256","true"],default="BW",help='''The Output Color: 
    BW: Black and White 
    grey: Greyscale 
    red: Redscale 
    green: Greenscale 
    blue: Bluescale 
    16: 16 Color ansi palette 
    256: 256 Color ansi palette 
    true: Same color scheme as the original
    (default: grey)
   ''') 
    parser.add_argument("--background-color",type=rgb_color,help="The Background Color: (Default: 0,0,0)",default="0,0,0")
    parser.add_argument("--no-audio", help="Do not transcode the audio.", action=argparse.BooleanOptionalAction)
    parser.add_argument("--no-subtitles", help="Do not transcode the subtitles.", action=argparse.BooleanOptionalAction)
    parser.add_argument("--fps",type=int,help="fps to transcode the video to (default: same as original)")
    parser.add_argument("--characters",help="Characters to be used. (Default: all keyboard characters)",default= string.ascii_letters + string.digits + string.punctuation + " ")
    parser.add_argument("--show-ffmpeg",help="Show the ffmpeg output.", action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    CHARACTERS = list(args.characters)
    OUTPUT_WIDTH = args.resolution[1]
    OUTPUT_HEIGHT = args.resolution[0]
    if os.path.exists(args.output_path):
        print("Error: Output file already exists")
        exit(1)
    
    if not args.output_path.endswith(".gif") and not args.output_path.endswith(".mkv"):
        print("Error: Output file must either be a gif or an mkv file")
        exit(1)

    if not os.path.exists(args.font_path):
        print("Error: Could not open font file")
        exit(1)

    rows, cols, font_size = find_optimal_font_size(args.font_path, OUTPUT_HEIGHT,OUTPUT_WIDTH,CHARACTERS,args.min_font_size,args.max_font_size)
    if rows is None:
        print("No acceptable font size found in range for video size and range")
        exit(1)
    font = ImageFont.truetype(args.font_path, font_size)
    character_cache = generate_character_cache(font,CHARACTERS)
    input_video = cv2.VideoCapture(args.video_path) 
    if not input_video.isOpened():
        print("Error: Could not open video file.")
        exit(1)

    total_frames = input_video.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = input_video.get(cv2.CAP_PROP_FPS)
    font_width, font_height = calculate_font_dimensions(font,CHARACTERS)
    transformer = partial(transform_frame, font=font, rows=rows, cols=cols, characters=character_cache,font_width=font_width,font_height=font_height, args=args)
    input_images = video_frame_iterator(input_video)

    pipe_input = ffmpeg.input('pipe:', format='rawvideo', pix_fmt='bgr24', s=f'{OUTPUT_WIDTH}x{OUTPUT_HEIGHT}', r=fps) #Input for the piped in ascii frames
    video_input = ffmpeg.input(args.video_path)
    output_streams = []
    if args.fps is None:
        output_streams.append(pipe_input.video)
    else:
        output_streams.append(ffmpeg.filter(pipe_input.video, 'fps', fps=args.fps))
    video_input = ffmpeg.input(args.video_path)
    if not args.no_audio and not args.output_path.endswith(".gif"):
        output_streams.append(video_input.audio)

    if not args.no_subtitles and not args.output_path.endswith(".gif"):
        output_streams.append(video_input["s?"])    

    vcodec = "gif" if args.output_path.endswith(".gif") else "libx264"
    process = (
        ffmpeg.output(
            *output_streams,
            args.output_path,
            vcodec=vcodec
        )
        .run_async(pipe_stdin=True,quiet=not args.show_ffmpeg)
    )
    pool = Pool()
    try:
        for frame in tqdm(pool.imap(transformer, input_images, chunksize=1), total=total_frames, desc="Processing frames", unit=" frames"):
            process.stdin.write(frame.tobytes())
            process.stdin.flush() # limits frame processing to the speed of the ffmpeg process, avoids overfilling ram with queued frames

        pool.close()
        pool.join()
        process.stdin.close()
        process.wait()
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        process.terminate()
        pool.terminate()
        input_video.release()
