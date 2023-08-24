from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget, QFileDialog, QTextEdit
from PySide6.QtGui import QPixmap, QImage
import sys
from tensorflow.keras.utils  import load_img
from keras.applications.resnet import preprocess_input
from keras.applications.resnet50 import ResNet50, decode_predictions
from tensorflow.keras.utils import img_to_array
from keras.models import load_model
from keras.preprocessing import image
import numpy as np

class MyWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dog breed recognize")

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
        model = ResNet50(weights='imagenet')
        if self.image_path:
           img = image.load_img(self.image_path, target_size=(224, 224))
           x = image.img_to_array(img)
           x = np.expand_dims(x, axis=0)
           x = preprocess_input(x)

           preds = model.predict(x)

           print('Predicted:', decode_predictions(preds, top=3)[0])

           html = decode_predictions(preds, top=3)[0]
           res = []
           for e in html:
               res.append((e[1], np.round(e[2] * 100, 2)))
           print(res)
           text = ""
           for key, value in res:
               text += f"{key}: {value}\n"
           print(text)
           self.textEdit.setPlainText(text)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyWindow()
    window.show()
    sys.exit(app.exec_())
