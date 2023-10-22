import cv2
import qrcode
import sys
from PyQt5.QtWidgets import *
from PyQt5 import QtGui, uic

class MyGui(QMainWindow):

    def __init__(self):
        super(MyGui, self).__init__()
        uic.loadUi("QRCODE.ui", self)
        self.show()

        self.current_file = ""
        self.actionLoad.triggered.connect(self.load_image)
        self.actionSave.triggered.connect(self.save_image)
        self.actionQuit.triggered.connect(self.quite)
        self.pushButton.clicked.connect(self.generate_code)
        self.pushButton_2.clicked.connect(self.read_code)

    def load_image(self):
        options = QFileDialog.Options()
        filename, _ = QFileDialog.getOpenFileName(self, "Open File", "", "All Files(*)", options=options)
        if filename != "":
            self.current_file = filename
            pixmap = QtGui.QPixmap(self.current_file)
            pixmap = pixmap.scaled(300, 300)
            self.label.setScaledContents(True)
            self.label.setPixmap(pixmap)

    def save_image(self):
        options= QFileDialog.Options()
        filename,_=QFileDialog.getSaveFileName(self,"Save File", "","JPG(*.jpg)", options=options)
        if filename != "":
            img=self.label.pixmap()
            img.save(filename,"JPG")

    def generate_code(self):
        qr = qrcode.QRCode(version=1,
                           error_correction=qrcode.constants.ERROR_CORRECT_L,
                           box_size=20,
                           border=2)
        qr.add_data(self.textEdit.toPlainText())
        qr.make(fit=True)
        img=qr.make_image(fill_color='blue', back_color='green')
        img.save("currentqr.jpg")

        pixmap=QtGui.QPixmap('currentqr.jpg')
        pixmap=pixmap.scaled(300,300)
        self.label.setScaledContents(True)
        self.label.setPixmap(pixmap)

    def quite(self):
        sys.exit(0)

    def read_code(self):
        try:
            img = cv2.imread(self.current_file)
            if img is not None:
               detector=cv2.QRCodeDetector()
               data,_,_=detector.detectAndDecode(img)
               self.textEdit.setText(data)

            else:
                print("Failed to open the image.")
        except Exception as e:
            print(f"An error occurred: {str(e)}")


def main():
    app = QApplication([])
    window=MyGui()
    app.exec()


if __name__ == "__main__":
    main()
