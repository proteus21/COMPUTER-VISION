import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog, QLabel
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
import sys

class FaceRecognitionApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.init_ui()

    def init_ui(self):
        self.setGeometry(100, 100, 800, 600)
        self.setWindowTitle("Face Recognition number App")

        self.status_label = QLabel("No faces detected", self)
        self.statusBar().addWidget(self.status_label)

        self.image_label = QLabel(self)
        self.image_label.setGeometry(10, 50, 600, 400)
        self.image_label.setAlignment(Qt.AlignCenter)

        upload_button = QPushButton("Upload Image", self)
        upload_button.setGeometry(10, 10, 150, 30)
        upload_button.clicked.connect(self.upload_image)
        self.image_path = None

        video_button = QPushButton("Open Video", self)
        video_button.setGeometry(170, 10, 150, 30)
        video_button.clicked.connect(self.open_video)

        self.video_capture = None

    def image_face_recognize(self, image):
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        self.status_label.setText(f"Number of faces detected: {len(faces)}")

        i = 0

        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.putText(image, f'Face {i + 1}', (x - 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            i += 1

        # Scale the image to fit within the label while maintaining the aspect ratio
        q_image = self.scale_image(image, self.image_label.size())
        self.image_label.setPixmap(QPixmap.fromImage(q_image))

    def scale_image(self, image, label_size):
        img_height, img_width = image.shape[:2]
        label_width, label_height = label_size.width(), label_size.height()

        if img_width > img_height:
            scaled_width = label_width
            scaled_height = int(img_height * (label_width / img_width))
        else:
            scaled_height = label_height
            scaled_width = int(img_width * (label_height / img_height))

        scaled_image = cv2.resize(image, (scaled_width, scaled_height))
        q_image = QImage(scaled_image.data, scaled_width, scaled_height, scaled_image.strides[0], QImage.Format_RGB888)

        return q_image

    def upload_image(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getOpenFileName(self, "Choose image", "", "Images (*.png *.jpg *.bmp *.gif)", options=options)

        if file_name:
            image = cv2.imread(file_name)
            self.image_face_recognize(image)

    def open_video(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        video_path, _ = QFileDialog.getOpenFileName(self, "Open Video", "", "Video Files (*.mp4 *.avi)", options=options)

        if video_path:
            self.video_capture = cv2.VideoCapture(video_path)
            self.process_video()

    def process_video(self):
        if self.video_capture.isOpened():
            ret, frame = self.video_capture.read()
            if ret:
                self.image_face_recognize(frame)
                cv2.waitKey(30)  # Adjust the wait time between frames
                self.process_video()

    def closeEvent(self, event):
        if self.video_capture:
            self.video_capture.release()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FaceRecognitionApp()
    window.show()
    sys.exit(app.exec_())
