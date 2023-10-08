from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget, QFileDialog
from PySide6.QtWidgets import QTextEdit
from PySide6.QtGui import QPixmap, QImage
import sys
from deepface import DeepFace

class MyWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DeepFace - Age & Gender ")

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
        if self.image_path:
            try:
                results = DeepFace.analyze(self.image_path, actions=["gender", "age"])
                if 'gender' in results[0] and 'age' in results[0]:
                    gender = results[0]['gender']
                    age = results[0]['age']
                    result_text = "Gender: {}\nAge: {}".format(gender, age)
                    self.textEdit.setPlainText(result_text)
                else:
                    self.textEdit.setPlainText("Gender and age information not found.")
            except Exception as e:
                self.textEdit.setPlainText(str(e))




            except Exception as e:
                self.textEdit.setPlainText(str(e))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyWindow()
    window.show()
    sys.exit(app.exec_())