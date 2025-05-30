# ASCII Encoder

This Python script converts a video into an ASCII art video. it can read any format supported by opencv, (examples include: mkv, mp4, avi, and others) and can write to either mkv or gif. It supports copying audio and subtitles from the input stream. 

## usage

```
usage: asciiencoder.py [-h] [--resolution RESOLUTION] [--min-font-size MIN_FONT_SIZE] [--max-font-size MAX_FONT_SIZE]
                       [--color {BW,grey,red,green,blue,16,256,true}] [--background-color BACKGROUND_COLOR]
                       [--no-audio | --no-no-audio] [--no-subtitles | --no-no-subtitles] [--fps FPS] [--characters CHARACTERS]
                       [--show-ffmpeg | --no-show-ffmpeg]
                       video_path font_path output_path

convert videos into ascii movies

positional arguments:
  video_path            Path to the video file.
  font_path             Path to a ttf or otf font file. (Monospace fonts are recommended)
  output_path           Path to save the ASCII movie.

options:
  -h, --help            show this help message and exit
  --resolution RESOLUTION
                        Resolution of the ASCII movie. (Default: 720x1280)
  --min-font-size MIN_FONT_SIZE
                        Minimum font size in pixels. (Default: 8)
  --max-font-size MAX_FONT_SIZE
                        Maximum font size in pixels. (Default: 24)
  --color {BW,grey,red,green,blue,16,256,true}
                        The Output Color:
                            BW: Black and White
                            grey: Greyscale
                            red: Redscale
                            green: Greenscale
                            blue: Bluescale
                            16: 16 Color ansi palette
                            256: 256 Color ansi palette
                            true: Same color scheme as the original
                            (default: grey)

  --background-color BACKGROUND_COLOR
                        The Background Color: (Default: 0,0,0)
  --no-audio, --no-no-audio
                        Do not transcode the audio.
  --no-subtitles, --no-no-subtitles
                        Do not transcode the subtitles.
  --fps FPS             fps to transcode the video to (default: same as original)
  --characters CHARACTERS
                        Characters to be used. (Default: all keyboard characters)
  --show-ffmpeg, --no-show-ffmpeg
                        Show the ffmpeg output.
```

## Requirements

this program requires that ffmpeg be on your computer and in your path, you must also install all of the libraries in the requirements.txt file
this can be done with the command ```pip install -r requirements.txt```

## Examples

Examples of the output are given below, Example gifs are large and may take a minute to load on slower connections. they can be found in the examples folder

### Shrek


```python asciiencoder.py input/shrek.mkv cour.ttf examples/shrek.gif --color true --resolution 360x640 --fps 5```

![Shrek](examples/shrek.gif)

### The Lion King

```python asciiencoder.py input/lionking.mp4 cour.ttf examples/lionking.gif --color 16 --resolution 360x640 --fps 5```

![The Matrix](examples/lionking.gif)

### The Matrix

```python asciiencoder.py input/matrix.mkv cour.ttf examples/matrix.gif --color green --resolution 360x640 --fps 5```

![The Matrix](examples/matrix.gif)


