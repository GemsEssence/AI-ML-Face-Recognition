# Face Detection with Python

This project is a Python-based face detection tool that uses the `dlib`, `face-recognition`, and `OpenCV` libraries. It detects faces in images or video feeds and can identify known faces by comparing them with pre-trained facial data.

## Features

- Detects and recognizes faces from images or real-time video feed.
- Identifies known faces by comparing them with saved facial encodings.
- Adjustable to handle different face recognition scenarios, such as single or multiple face detection.

## Requirements

The following dependencies are required to run this project. Install them using the provided `requirements.txt` file:

```plaintext
click==8.1.7
dlib==19.24.6
face-recognition==1.3.0
face_recognition_models==0.3.0
numpy==2.1.3
opencv-python==4.10.0.84
pillow==11.0.0
```

## Installation

1. **Clone this repository**:
   ```bash
   git clone https://github.com/your-username/face-detection
   cd face-detection
   ```

2. **Create and activate a virtual environment**:

   - **On Windows**:
     ```bash
     python -m venv venv
     .\venv\Scripts\activate
     ```

   - **On macOS and Linux**:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

3. **Install the dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the face detection script with the following command:

```bash
python face_detection.py
```

This command will start the script, which can be configured to detect faces from an image file or a video feed.

## Project Structure

- **face_detection.py**: Main script for face detection and recognition.
- **requirements.txt**: Lists all required Python libraries.
- **train**: Directory containing all training images.

