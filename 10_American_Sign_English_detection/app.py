from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget, QFileDialog, QTextEdit
from PySide6.QtGui import QPixmap, QImage
import sys
from tensorflow.keras.utils  import load_img
from tensorflow.keras.utils import img_to_array
from keras.models import load_model
from keras.preprocessing import image
import numpy as np

class MyWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SNL recognize")

        self.uploadButton = QPushButton("Upload image")
        self.uploadButton.clicked.connect(self.upload_image)

        self.recognizeButton = QPushButton("Recognize")
        self.recognizeButton.clicked.connect(self.detect_object)

        self.label = QLabel(self)
        self.setCentralWidget(self.label)

        self.textEdit = QTextEdit(self)
        self.textEdit.setReadOnly(True)

        layout = QVBoxLayout()
        layout.addWidget(self.uploadButton)
        layout.addWidget(self.label)
        layout.addWidget(self.recognizeButton)
        layout.addWidget(self.textEdit)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.image_path = None

    def upload_image(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getOpenFileName(self, "Choose image", "", "Images (*.png *.jpg *.bmp *.gif)", options=options)

        if file_name:
            self.image_path = file_name
            pixmap = QPixmap(self.image_path)
            self.label.setPixmap(pixmap)


    def detect_object(self):
        model_path = 'C:/Users/prote/PycharmProjects/SNL/venv/ASl.h5'
        model = load_model(model_path)
        class_names=['A', 'B', 'Blank', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
        if self.image_path:
           img = image.load_img(self.image_path, target_size=(128, 128))
           x = image.img_to_array(img)
           x = np.expand_dims(x, axis=0)
           preds = model.predict(x)
           print('Predicted:', class_names[np.argmax(preds)])

           self.textEdit.setPlainText(f'Predicted:{class_names[np.argmax(preds)]}')

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyWindow()
    window.show()
    sys.exit(app.exec())