# Face Detection with Python

This project is a Python-based face detection tool that uses the `dlib`, `face-recognition`, and `OpenCV` libraries. It detects faces in images or video feeds and can identify known faces by comparing them with pre-trained facial data.

## Features

- Detects and recognizes faces from images or real-time video feed.
- Identifies known faces by comparing them with saved facial encodings.
- Adjustable to handle different face recognition scenarios, such as single or multiple face detection.

## Installation

1. **Clone this repository**:
   ```bash
   git clone git@github.com:GemsEssence/AI-ML-Face-Recognition-.git
   ```

2. **Create and activate a virtual environment**:

   - **On Windows**:
     ```bash
     python -m venv env
     env\Scripts\activate.bat
     ```

   - **On macOS and Linux**:
     ```bash
     python3 -m venv env
     source env/bin/activate
     ```

3. **Install the dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Add face images to the **train** directory for the individuals you want to detect. 
2. Run the face detection script with the following command:

    ```bash
    python face_detection.py
    ```
    
    This command will start the script, which can be configured to detect faces from an image file or a video feed.
3. Press "**q**" key to close the program.

## Project Structure

- **face_detection.py**: Main script for face detection and recognition.
- **requirements.txt**: Lists all required Python libraries.
- **train**: Directory containing all training images.

