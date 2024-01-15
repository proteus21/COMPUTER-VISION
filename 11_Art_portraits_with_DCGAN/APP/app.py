from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
from model import generate
import tensorflow as tf

app = Flask(__name__)
img_rows,img_cols, channels =128,128,3
UPLOAD_FOLDER= 'static'
ALLOWED_EXTENSIONS=set(['jpg','png'])
app.config['UPLOAD_FOLDER']=UPLOAD_FOLDER


@app.route('/')
def start():
    return  render_template('index.html')

@app.route('/', methods=['GET','POST'])
def upload():
    if request.method=='POST':
        path=os.path.join(app.config['UPLOAD_FOLDER'],'input.png')
        plt.switch_backend('Agg')
        image=generate()
        image=image.reshape((img_rows,img_cols, channels)).astype('uint8')
        image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)
        plt.imshow(image)
        plt.axis('off')
        plt.imsave(path, image)
    return render_template('index.html')



if __name__ == '__main__':
    app.run(DEBUG=True)
