import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
import keras

generator=keras.models.load_model('generator_1.h5')

noise_dim=100

def generate():
    noise=np.random.normal(0,1,size=(1,noise_dim))
    generated_images=generator.predict(noise)
    generated_images=generated_images*127.5+127.5
    for  i, image in enumerate(generated_images):
        img=image
    return img
