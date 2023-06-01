# Face Age Detection

This Python script uses OpenCV and trained Deep Learning models to identify faces in a video stream (from a webcam or a video file) and categorize the detected faces into three age groups: Under 18, Check ID, and Over 18. The script then highlights the faces with color codes depending on the age group.

## Functionality
The function `highlight_face` processes frames from a video stream and detects faces. It uses a trained neural network (loaded in the variable `net`) to perform face detection, and returns a list of bounding boxes for the detected faces.

The main script reads video and image data from a source specified as a command-line argument (`--source`). If no source is specified, it defaults to the webcam (`0`).

## Usage
To use the script, you need to have the trained models and the script in your working directory.

Run the script using the command line. The script accepts command line arguments to specify the source of the video and whether to allow more than one face per frame.

```
python3 detect.py --source 'path_to_video_file' --one_face
```

- `--source` argument: replace `'path_to_video_file'` with the path to a video file you want to use. OpenCV can handle a variety of formats such as .mp4 and .jpg. If `--source` is not specified, the script defaults to the webcam.
- `--one_face` argument: include this argument if you want the script to process frames with only one face. If this argument is not included, the script processes all frames regardless of the number of faces.

## Dependencies
The script requires OpenCV and NumPy to be installed in your Python environment. You can install these packages using pip:

```
pip install opencv-python numpy
```

Also, the script uses pretrained models for face detection and age categorization. Make sure you have these model files in your working directory:

- `opencv_face_detector.pbtxt`
- `opencv_face_detector_uint8.pb`
- `age_deploy.prototxt`
- `age_net.caffemodel`

