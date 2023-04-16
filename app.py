from flask import Flask, render_template, request, redirect, url_for
from flask_cors import CORS,cross_origin
import time 
import cv2 as cv
import numpy as np
from PIL import Image


app = Flask(__name__)

isAuthenticated=False 

@app.route('/')
def login_page():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST': 
        username = request.form['username']
        password = request.form['password']
        print(username,password)
        if username == 'abc' and password == '123':   
           isAuthenticated=True
           return redirect(url_for('predict_page'))
    return render_template('login.html')
    
# Define the function that calculates the time limits per vehicle
def calculate_time_limits(vehicles, base_timer, time_limits):
    time_limits_per_vehicle = []
    for vehicle, quantity in vehicles.items():
        time_limit = (quantity / sum(vehicles.values())) * base_timer
        if time_limits[0] < time_limit < time_limits[1]:
            time_limits_per_vehicle.append(time_limit)
        else:
            closest_limit = min(time_limits, key=lambda x: abs(x - time_limit))
            time_limits_per_vehicle.append(closest_limit)
    return time_limits_per_vehicle, sum(time_limits_per_vehicle)


@app.route('/predict', methods=['GET', 'POST'])
def predict_page():
    if request.method == 'POST':
        # Get input values from the form
        signal1 = int(request.form['signal1'])
        signal2 = int(request.form['signal2'])
        signal3 = int(request.form['signal3'])
        signal4 = int(request.form['signal4'])
        # Define the inputs for the time limit calculation function
        vehicles = {'vehicle1': signal1, 'vehicle2': signal2, 'vehicle3': signal3, 'vehicle4': signal4}
        base_timer = 120
        time_limits = [5, 40]
        # Calculate the time limits per vehicle
        time_limits_per_vehicle, time_limits_sum = calculate_time_limits(vehicles, base_timer, time_limits)
        return render_template('Predict.html', time_limits_per_vehicle=time_limits_per_vehicle, time_limits_sum=time_limits_sum)
    # Render the template with default values
    return render_template('Predict.html', time_limits_per_vehicle=[], time_limits_sum=0)

@app.route('/index',methods=['GET','POST']) 
def vehicle_cnt():
    return render_template('index.html') 

def process_images(img1, img2, img3, img4):
    classes = open('coco.names').read().strip().split('\n')
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(len(classes), 3), dtype='uint8')
    # Give the configuration and weight files for the model and load the network.
    net = cv.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
    # determine the output layer
    ln = net.getLayerNames()
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
    cars1, cars2, cars3, cars4=0,0,0,0
    blob1 = cv.dnn.blobFromImage(img1, 1/255.0, (416, 416), swapRB=True, crop=False)
    blob2 = cv.dnn.blobFromImage(img2, 1/255.0, (416, 416), swapRB=True, crop=False)
    blob3 = cv.dnn.blobFromImage(img3, 1/255.0, (416, 416), swapRB=True, crop=False)
    blob4 = cv.dnn.blobFromImage(img4, 1/255.0, (416, 416), swapRB=True, crop=False)

    net.setInput(blob1)
    outputs1 = net.forward(ln)
    net.setInput(blob2)
    outputs2 = net.forward(ln)
    net.setInput(blob3)
    outputs3 = net.forward(ln)
    net.setInput(blob4)
    outputs4 = net.forward(ln)

    cars1 = 0
    cars2 = 0
    cars3 = 0
    cars4 = 0

    boxes = []
    confidences = []
    classIDs = []
    h, w = img1.shape[:2]

    for output in outputs1:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            if classes[classID] in  ['car', 'motorbike', 'bus','truck']:
                confidence = scores[classID]
                if confidence > 0.5:
                    box = detection[:4] * np.array([w, h, w, h])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    box = [x, y, int(width), int(height)]
                    boxes.append(box)
                    confidences.append(float(confidence))
                    classIDs.append(classID)

    indices = cv.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    if len(indices) > 0:
        for i in indices.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            color = [int(c) for c in colors[classIDs[i]]]
            cv.rectangle(img1, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(classes[classIDs[i]], confidences[i])
            cars1+=1
            cv.putText(img1, text, (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1) 

    boxes = []
    confidences = []
    classIDs = []
    h, w = img2.shape[:2]
    for output in outputs2:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            if classes[classID] in  ['car', 'motorbike', 'bus','truck']:
                confidence = scores[classID]
                if confidence > 0.5:
                    box = detection[:4] * np.array([w, h, w, h])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    box = [x, y, int(width), int(height)]
                    boxes.append(box)
                    confidences.append(float(confidence))
                    classIDs.append(classID)

    indices = cv.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    if len(indices) > 0:
        for i in indices.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            color = [int(c) for c in colors[classIDs[i]]]
            cv.rectangle(img2, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(classes[classIDs[i]], confidences[i])
            cars2+=1
            cv.putText(img2, text, (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1) 

    boxes = []
    confidences = []
    classIDs = []
    h, w = img3.shape[:2]
    for output in outputs3:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            if classes[classID] in  ['car', 'motorbike', 'bus','truck']:
                confidence = scores[classID]
                if confidence > 0.5:
                    box = detection[:4] * np.array([w, h, w, h])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    box = [x, y, int(width), int(height)]
                    boxes.append(box)
                    confidences.append(float(confidence))
                    classIDs.append(classID)

    indices = cv.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    if len(indices) > 0:
        for i in indices.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            color = [int(c) for c in colors[classIDs[i]]]
            cv.rectangle(img3, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(classes[classIDs[i]], confidences[i])
            cars3+=1
            cv.putText(img3, text, (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1) 

    boxes = []
    confidences = []
    classIDs = []
    h, w = img4.shape[:2]
    for output in outputs4:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            if classes[classID] in  ['car', 'motorbike', 'bus','truck']:
                confidence = scores[classID]
                if confidence > 0.5:
                    box = detection[:4] * np.array([w, h, w, h])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    box = [x, y, int(width), int(height)]
                    boxes.append(box)
                    confidences.append(float(confidence))
                    classIDs.append(classID)

    indices = cv.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    if len(indices) > 0:
        for i in indices.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            color = [int(c) for c in colors[classIDs[i]]]
            cv.rectangle(img4, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(classes[classIDs[i]], confidences[i])
            cars4+=1
            cv.putText(img4, text, (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1) 

    print(cars1)
    print(cars2)
    print(cars3)
    print(cars4) 
    return 0

cnt=0

@app.route('/upload', methods=['POST','GET'])
def upload_images(): 
    global cnt 
    cnt+=1
    # imgIndex = 1
    # imgCount = 8
    signal1 = cv.imread(f'./static/{cnt}.jpg')
    signal2 = cv.imread(f'./static/{cnt+1}.jpg')
    signal3 = cv.imread(f'./static/{cnt+2}.jpg')
    signal4 = cv.imread(f'./static/{cnt+3}.jpg')
    # result = process_images(signal1,signal2,signal3,signal4)
    # print(result)
    return render_template('index.html',result=1,cnt=cnt)



if __name__ == "__main__":
    app.run(debug=True)