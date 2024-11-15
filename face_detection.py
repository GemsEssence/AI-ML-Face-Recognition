import os
import sys
import cv2
import face_recognition
import numpy as np
import datetime

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QIcon, QImage, QPixmap
from PyQt6.QtWidgets import QApplication, QMainWindow, QFileDialog

from face_detection_UI import Ui_Face_Detection  # Import the generated UI file


class Face_Detection(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()

        # Timer for updating frames
        self.main_widget_timer = QTimer(self)
        self.main_widget_timer.timeout.connect(self.update_frame)

        # Timer for updating camera frames
        self.image_widget_timer = QTimer(self)
        self.image_widget_timer.timeout.connect(self.update_camera_frame)
        
        # Video capture
        self.cap = cv2.VideoCapture(0)

        # Load known encodings
        self.images = []
        self.classNames = []
        self.uploaded_image = False
        self.captured_image = False
        self.load_images()
        self.encoded_face_train = self.findEncodings(self.images)

        self.start_webcam()
        self.start_main_widget_timer()


    def init_ui(self):
        # Set up the user interface from Designer.
        self.ui = Ui_Face_Detection()
        self.ui.setupUi(self)
        self.setup_connections()
        self.ui.stackedWidget.setCurrentWidget(self.ui.main_widget)


    def setup_connections(self):
        self.ui.add_prsn_btn.clicked.connect(self.image_widget)
        self.ui.back_btn.clicked.connect(self.main_widget)
        self.ui.img_upload_btn.clicked.connect(self.upload_image)
        self.ui.camera_btn.clicked.connect(self.start_camera)
        self.ui.capture_btn.clicked.connect(self.capture_photo)
        self.ui.submit_btn.clicked.connect(self.submit_image)


    def image_widget(self):
        self.stop_main_widget_timer()
        self.stop_webcam()
        self.ui.capture_btn.hide()
        self.ui.camera_btn.show()
        self.ui.invalid_img_msg.setHidden(True)
        self.ui.invalid_name_msg.setHidden(True)
        self.ui.display_image.clear()
        self.ui.name.clear()
        self.ui.stackedWidget.setCurrentWidget(self.ui.image_widget)


    def main_widget(self):
        self.stop_camera()
        self.start_webcam()
        self.start_main_widget_timer()
        self.ui.stackedWidget.setCurrentWidget(self.ui.main_widget)


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
                self.encoded_all_faces = False
                print("Face encoding failed for an image. Skipping this image.")
        return encodeList


    def start_webcam(self):
        self.cap.open(0)  # Open webcam


    def stop_webcam(self):
        self.cap.release()


    def start_main_widget_timer(self):
        self.main_widget_timer.start(30)  # Start updating frames


    def stop_main_widget_timer(self):
        self.main_widget_timer.stop()


    def start_image_widget_timer(self):
        self.image_widget_timer.start(30)  # Start updating frames


    def stop_image_widget_timer(self):
        self.image_widget_timer.stop()


    def start_camera(self):
        self.start_webcam()
        self.start_image_widget_timer()
        self.ui.capture_btn.show()
        self.ui.camera_btn.hide()
        self.ui.img_upload_btn.hide()


    def stop_camera(self):
        self.stop_webcam()
        self.stop_image_widget_timer()
        self.ui.capture_btn.hide()
        self.ui.camera_btn.show()
        self.ui.img_upload_btn.show()


    def upload_image(self):
        file_dialog = QFileDialog()
        self.file_path, _ = file_dialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        self.uploaded_image = False
        self.captured_image = False
        if self.file_path:
            # Resize the image to fit the dimensions of display_image
            display_size = self.ui.display_image.size()

            # Load and display the uploaded image
            self.img = cv2.imread(self.file_path)
            img_rgb = cv2.resize(self.img, (display_size.width(), display_size.height() - 100), interpolation=cv2.INTER_LINEAR)
            img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)

            # Display image in the QLabel
            height, width, channel = img_rgb.shape
            step = channel * width
            qImg = QImage(img_rgb.data, width, height, step, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(qImg)

            # Resize pixmap to fit QLabel while maintaining aspect ratio
            self.ui.display_image.setPixmap(pixmap.scaled(self.ui.display_image.size(), Qt.AspectRatioMode.KeepAspectRatio))
            self.uploaded_image = True


    def submit_image(self):
        self.ui.invalid_img_msg.setHidden(True)
        self.ui.invalid_name_msg.setHidden(True)

        if not (self.uploaded_image or self.captured_image):
            self.ui.invalid_img_msg.setHidden(False)
            return

        text_name = self.ui.name.text().strip().replace(" ", "")
        if not text_name:
            self.ui.invalid_name_msg.setText("Enter Name !!!")
            self.ui.invalid_name_msg.setHidden(False)
            return
        if text_name in self.classNames:
            self.ui.invalid_name_msg.setText("Name Already Taken")
            self.ui.invalid_name_msg.setHidden(False)
            return

        # Encode image and handle any potential errors
        self.encoded_all_faces = True
        try:
            encoded_face = face_recognition.face_encodings(cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB))[0]
        except IndexError:
            self.ui.invalid_img_msg.setHidden(False)
            return

        # Save the image and update training data
        save_path = os.path.join("train", f"{text_name}.jpg")
        cv2.imwrite(save_path, self.img)

        # Update known encodings and class names
        self.images.append(self.img)
        self.classNames.append(text_name)
        self.encoded_face_train.append(encoded_face)

        self.main_widget()
        self.uploaded_image = False
        self.captured_image = False


    def update_camera_frame(self, captured_photo=False):
        success, img = self.cap.read()

        if not success:
            return

        # Resize the image to fit the dimensions of display_image
        display_size = self.ui.display_image.size()
        img = cv2.resize(img, (display_size.width(), display_size.height() - 100), interpolation=cv2.INTER_LINEAR)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width, channel = img.shape
        step = channel * width
        qImg = QImage(img.data, width, height, step, QImage.Format.Format_RGB888)
        self.ui.display_image.setPixmap(QPixmap.fromImage(qImg))

        if captured_photo:
            self.img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


    def capture_photo(self):
        self.update_camera_frame(captured_photo=True)
        self.stop_camera()
        self.uploaded_image = False
        self.captured_image = True


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

        # Resize the image to fit the dimensions of video_label
        video_img_size = self.ui.video_label.size()
        img = cv2.resize(img, (video_img_size.width(), video_img_size.height() - 100), interpolation=cv2.INTER_LINEAR)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width, channel = img.shape
        step = channel * width
        qImg = QImage(img.data, width, height, step, QImage.Format.Format_RGB888)
        self.ui.video_label.setPixmap(QPixmap.fromImage(qImg))
        

    def closeEvent(self, event):
        self.stop_main_widget_timer()
        self.stop_webcam()
        cv2.destroyAllWindows()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    # app.setWindowIcon(QIcon('icons/app_icon.svg'))
    window = Face_Detection()
    window.show()
    sys.exit(app.exec())

