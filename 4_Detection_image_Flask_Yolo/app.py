import imghdr
import os
import cv2
import numpy as np

from flask import Flask, render_template, request, redirect, url_for, abort, \
    send_from_directory, session
from werkzeug.utils import secure_filename

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


# YOLO object detection function
def detect_object(uploaded_image_path):
    # Loading image
    img = cv2.imread(uploaded_image_path.replace('\\','/'))
    height, width, channels = img.shape

    # Load Yolo
    yolo_weight = "data/model/yolov3.weights"
    yolo_config = "data/model/yolov3.cfg"
    coco_labels = "data/model/coco.names"
    net = cv2.dnn.readNet(yolo_weight, yolo_config)

    classes = []
    with open(coco_labels, "r") as f:
        classes = [line.strip() for line in f.readlines()]

    # print(classes)

    # # Defining desired shape

    fWidth = width
    fHeight = height

    # Resize image in opencv
    img = cv2.resize(img, (fWidth, fHeight))

    height, width, channels = img.shape

    # Convert image to Blob
    blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416),(0,0,0),swapRB=True, crop=False)
    # Set input for YOLO object detection
    net.setInput(blob)

    # Find names of all layers
    layer_names = net.getLayerNames()
    # print(layer_names)
    # Find names of three output layers
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    print(output_layers)

    # Send blob data to forward pass
    outs = net.forward(output_layers)
    print(outs[0].shape)
    print(outs[1].shape)
    print(outs[2].shape)

    # Generating random color for all 80 classes
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    # Extract information on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            # Extract score value
            scores = detection[5:]
            # Object id
            class_id = np.argmax(scores)
            # Confidence score for each object ID
            confidence = scores[class_id]
            # if confidence > 0.5 and class_id == 0:
            if confidence > 0.5:
                # Extract values to draw bounding box
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    # Draw bounding box with text for each object
    font = cv2.FONT_HERSHEY_DUPLEX
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence_label = int(confidences[i] * 100)
            color = colors[i]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, f'{label, confidence_label}', (x - 25, y + 75), font, 1, color, 2)

    # Write output image (object detection output)
    output_image_path = os.path.join(app.config['UPLOAD_PATH'] , 'output_image.jpg')
    cv2.imwrite(output_image_path, img)

    return (output_image_path)


@app.route('/detect_object')
def detectObject():
    uploaded_image_path = session.get('uploaded_img_file_path', None)
    output_image_path = detect_object(uploaded_image_path)
    print(output_image_path)
    return render_template('show_image.html', user_image=output_image_path)


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