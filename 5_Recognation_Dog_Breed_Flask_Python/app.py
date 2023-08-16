import imghdr
import os
import cv2
import numpy as np
from tensorflow.keras.utils  import load_img
from keras.applications.resnet import preprocess_input
from keras.applications.resnet50 import ResNet50, decode_predictions
from tensorflow.keras.utils import img_to_array
from keras.models import load_model
from keras.preprocessing import image
from flask import Flask, render_template, request, redirect, url_for, abort, \
    send_from_directory, session
from werkzeug.utils import secure_filename
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

app.config['MAX_CONTENT_LENGTH'] =4 * 1024 * 1024
app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.png', '.gif']
app.config['UPLOAD_PATH'] = 'uploads'
app.secret_key = 'You Will Never Guess What is that'
def validate_image(stream):
    header = stream.read(512)  # 512 bytes should be enough for a header check
    stream.seek(0)  # reset stream pointer
    format = imghdr.what(None, header)
    if not format:
        return None
    return '.' + (format if format != 'jpeg' else 'jpg')

@app.route('/')
def index():
    files = os.listdir(app.config['UPLOAD_PATH'])
    return render_template('index.html', files=files)

@app.route('/', methods=['POST'])
def upload_files():
    uploaded_file = request.files['file']
    filename = secure_filename(uploaded_file.filename)
    if filename != '':
        file_ext = os.path.splitext(filename)[1]
        if file_ext not in app.config['UPLOAD_EXTENSIONS'] or \
                file_ext != validate_image(uploaded_file.stream):
            abort(400)
        uploaded_file.save(os.path.join(app.config['UPLOAD_PATH'], filename))
        session['uploaded_img_file_path'] = os.path.join(app.config['UPLOAD_PATH'],filename)
        print(session['uploaded_img_file_path'] )
    return redirect(url_for('index'))

@app.route('/uploads/<filename>')
def upload(filename):
    return send_from_directory(app.config['UPLOAD_PATH'], filename)

def detect_object(uploaded_image_path):
    model=ResNet50(weights='imagenet')
    img=image.load_img(uploaded_image_path.replace('\\','/'), target_size=(224,224))
    x=image.img_to_array(img)
    x=np.expand_dims(x,axis=0)
    x=preprocess_input(x)

    preds=model.predict(x)

    print('Predicted:',decode_predictions(preds, top=3)[0])

    html=decode_predictions(preds, top=3)[0]
    res=[]
    for e in html:
        res.append((e[1], np.round(e[2]*100,2)))
    print({'res':res})
    return res


def detect_object_1(uploaded_image_path):
    # Loading image
    input_dim = (224, 224)
    image = load_img(uploaded_image_path.replace('\\','/'), target_size=input_dim)
    print(image)
    image = img_to_array(image)
    image = image.reshape((1, *image.shape))
    image = preprocess_input(image)
    Last_model = load_model('data/model/model_3.h5')
    res = Last_model.predict(image)
    s = np.argsort(res)[0][-5:]
    labelE=LabelEncoder()
    labelE.fit_transform(s)
    labelE.inverse_transform(s)
    breed=labelE.inverse_transform(s)
    print(breed)
    return s,breed


@app.route('/detect_object')
def detectObject():
    uploaded_image_path = session.get('uploaded_img_file_path', None)
    res = detect_object(uploaded_image_path)
    user_image=uploaded_image_path

    return render_template('detect_image.html',res=res,user_image=uploaded_image_path)


# flask clear browser cache (disable cache)
# Solve flask cache images issue

@app.route('/show_image')
def displayImage():
    img_file_path = session.get('uploaded_img_file_path', None)
    return render_template('show_image.html', user_image=img_file_path)


@app.after_request
def add_header(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    response.headers['Cache-Control'] = 'public, max-age=0'
    return response