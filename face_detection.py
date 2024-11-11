import os
import sys
import cv2
import face_recognition
import numpy as np

from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QIcon, QImage, QPixmap
from PyQt6.QtWidgets import QApplication, QMainWindow, QFileDialog

from face_detection_UI import Ui_Face_Detection  # Import the generated UI file


class Face_Detection(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()

        # Timer for updating frames
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        
        # Video capture
        self.cap = cv2.VideoCapture(0)

        # Load known encodings
        self.images = []
        self.classNames = []
        self.load_images()
        self.encoded_face_train = self.findEncodings(self.images)

        self.start_webcam()


    def init_ui(self):
        # Set up the user interface from Designer.
        self.ui = Ui_Face_Detection()
        self.ui.setupUi(self)
        self.setup_connections()
        self.ui.stackedWidget.setCurrentWidget(self.ui.main)


    def setup_connections(self):
        self.ui.add_prsn_btn.clicked.connect(self.upload_image)


    def load_images(self):
        relative_path = 'train'
        path = os.path.abspath(relative_path)
        if not os.path.exists(path):
            os.makedirs(path)  # Create train directory if it doesn't exist
        mylist = os.listdir(path)
        for cl in mylist:
            curImg = cv2.imread(f'{path}/{cl}')
            self.images.append(curImg)
            self.classNames.append(os.path.splitext(cl)[0])


    def findEncodings(self, images):
        encodeList = []
        for img in images:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            try:
                encoded_face = face_recognition.face_encodings(img)[0]
                encodeList.append(encoded_face)
            except IndexError:
                print("Face encoding failed for an image. Skipping this image.")
        return encodeList


    def start_webcam(self):
        self.cap.open(0)  # Open webcam
        self.timer.start(30)  # Start updating frames


    def stop_webcam(self):
        self.timer.stop()
        self.cap.release()


    def upload_image(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if file_path:
            # Load and display the uploaded image
            img = cv2.imread(file_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Save the image to the "train" directory
            save_path = os.path.join("train", os.path.basename(file_path))
            cv2.imwrite(save_path, img)
            print(f"Image saved to {save_path}")

            # Add the new image to the training data
            self.images.append(img)
            self.classNames.append(os.path.splitext(os.path.basename(file_path))[0])
            self.encoded_face_train = self.findEncodings(self.images)  # Recalculate encodings

            # Display image in the QLabel
            height, width, channel = img_rgb.shape
            step = channel * width
            qImg = QImage(img_rgb.data, width, height, step, QImage.Format.Format_RGB888)
            self.ui.video_label.setPixmap(QPixmap.fromImage(qImg))


    def update_frame(self):
        success, img = self.cap.read()
        if not success:
            return

        imgS = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
        faces_in_frame = face_recognition.face_locations(imgS)
        encoded_faces = face_recognition.face_encodings(imgS, faces_in_frame)

        for encode_face, faceloc in zip(encoded_faces, faces_in_frame):
            faceDist = face_recognition.face_distance(self.encoded_face_train, encode_face)
            matches = face_recognition.compare_faces(self.encoded_face_train, encode_face, tolerance=0.5)

            try:
                matchIndex = np.argmin(faceDist)
                threshold = 0.9

                if faceDist[matchIndex] < threshold and matches[matchIndex]:
                    name = self.classNames[matchIndex].capitalize()
                else:
                    name = "Unknown Person"

            except ValueError as e:
                name = "Unknown Person"

            y1, x2, y2, x1 = faceloc
            y1, x2, y2, x1 = y1 * 2, x2 * 2, y2 * 2, x1 * 2
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 5), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width, channel = img.shape
        step = channel * width
        qImg = QImage(img.data, width, height, step, QImage.Format.Format_RGB888)
        self.ui.video_label.setPixmap(QPixmap.fromImage(qImg))
        

    def closeEvent(self, event):
        self.stop_webcam()
        cv2.destroyAllWindows()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    # app.setWindowIcon(QIcon('icons/app_icon.svg'))
    window = Face_Detection()
    window.show()
    sys.exit(app.exec())

