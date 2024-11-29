# YOLOv8 Video Tracking Application

This is a Flask application for video tracking using YOLOv8, specifically designed to detect and track people in videos. The application processes uploaded videos, applies YOLOv8 object detection to track the "person" class, and provides metrics like frames per second (FPS) for video processing.

## Features
- Upload a video file for processing.
- Track persons in the video using YOLOv8.
- Download the processed video with tracked persons.
- Display metrics such as FPS and processing time.

## Prerequisites
- Python 3.x
- Virtual environment (recommended)

## Installation

###  Clone the repository

```bash
git clone https://github.com/your-repository-name/yolov8-video-tracking.git
cd yolov8-video-tracking
```

### Install the required Python libraries using the requirements.txt file:

```bash
pip install -r requirements.txt
```
### Set up environment variables
#### .env file in the root directory of the project modify it if you need
- UPLOAD_FOLDER: Directory where uploaded videos will be saved.
- OUTPUT_FOLDER: Directory where processed videos will be saved.
- ALLOWED_EXTENSIONS: Comma-separated list of allowed video file extensions.
- YOLO_MODEL_PATH: Path to the YOLOv8 model weights (e.g., yolov8l.pt).
- CONFIDENCE_THRESHOLD: Confidence threshold for object detection.
- IOU_THRESHOLD: Intersection over Union (IoU) threshold for tracking.

### Run the application Gui
#### Once the setup is complete, run the Flask application with:

```bash
python app.py
```
### Run the api

```bash
python api.py
```
#### Test the API:
- You can use Postman or curl to test the API, just like before.
- Upload video (POST request):
```bash
curl -X POST -F "video=@path_to_video.mp4" http://127.0.0.1:5000/api/upload
```

#### Download processed video (GET request):
```bash
curl -O http://127.0.0.1:5000/api/download/output_video.avi

```
## Folder Structure
```bash
yolov8-video-tracking/
├── app.py
├── .env
├── requirements.txt
├── uploads/
├── output
    └──runs/
│       └── detect/
│       └── track/
├── templates/
│   └── index.html
└── models
```

