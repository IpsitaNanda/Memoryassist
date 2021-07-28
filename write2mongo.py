#import numpy as np
from flask import Flask, request, jsonify, render_template, url_for
import cv2
import os
import time
import numpy as np
import platform
import pymongo
from PIL import Image
#breakpoint()
import io
'''#import matplotlib.pyplot as plt
from bson.binary import Binary
from PIL import Image'''

def show_window():
    if platform.system() == "Windows" or platform.system() == "Darwin":
        from PIL import ImageGrab  # WINDOWS/MAC
    else:
        import pyscreenshot as ImageGrab  # LINUX

    width, height = (1600, 900)

def process_frame(frame, info, count, num_pic):
    # global count

    # Convert to grayscale (black and white)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Finding Faces
    faces = face_cascade.detectMultiScale(gray)

    num_faces = len(faces)

    # Give Message if No Faces are Detected
    if num_faces == 0:
        text1 = "No Face Detected!"
        text2 = "Please make sure " + info["Name"]+ " is in frame."
        cv2.putText(frame, text1, (0, 20), font, font_size, red, 2, cv2.LINE_AA)
        cv2.putText(frame, text2, (0, 40), font, font_size, red, 2, cv2.LINE_AA)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + h]

        # Give Message and Pause Data Collection if Multiple Faces are Detected
        if num_faces == 1:
            count += 1
            #file = info["Directory"] + "/" + str(count) + ".png"
            #cv2.imwrite(file, roi_color)
            mongodb_con(count, info,roi_color)
        else:
            text1 = str(num_faces) + " Faces Detected! Data Collection Paused."
            text2 = "Please make sure only " + info["Name"] + " is in frame."
            cv2.putText(frame, text1, (0, 20), font, font_size, red, 2, cv2.LINE_AA)
            cv2.putText(frame, text2, (0, 40), font, font_size, red, 2, cv2.LINE_AA)

        global text
        # Display count
        text = str(count) + "/" + str(num_pics)
        cv2.putText(frame, text, (x, y), font, font_size, white, 2, cv2.LINE_AA)

        # Draw Rectangle around face
        x_end = x + w
        y_end = y + h
        cv2.rectangle(frame, (x, y), (x_end, y_end), cyan, 3)
    return frame, count

def mongodb_con(pic_nb, info,roi_color):
    print("connecting to database...")
    client = pymongo.MongoClient(
        "mongodb://m001-student:Password@cluster0-shard-00-00.nvnbz.mongodb.net:27017/Positive?ssl=true&replicaSet"
        "=atlas-qi9ptr-shard-0&authSource=admin&retryWrites=true&w=majority")  # open connection to mongo db to save
    # face to database
    face_database = client[info["Name"]]  # select database
    faces_collection = face_database[info["PatientName"]]  # select collection
    img_path = info["Directory"] + "/" + str(pic_nb) + ".png"
    # return face_database,faces_collection
    print("connected to Mongodb")

    im = Image.fromarray(roi_color, 'RGB')
    image_bytes = io.BytesIO()
    im.save(image_bytes, format='PNG')
    profile = {
        'name': info['Name'],
        'patient_name': info['PatientName'],
        'relationship': info['Relationship'],
        'data': image_bytes.getvalue()
    }
    faces_collection.insert_one(profile).inserted_id
    print("image " + str(pic_nb) + ".png has been inserted")
    #os.remove(img_path)

def input_method():
    face_input = "webcam"
    #input("Record from screen or webcam: ").lower()
    if "screen" in face_input or "display" in face_input:
        return "screen"
    elif "cam" in face_input:
        return "webcam"
    #else:
    #    return input_method()

app = Flask(__name__)
@app.route('/')
def home():
    #return 'Hello World'
    return "HOME"
    #return render_template('index.html')

@app.route('/addfaces_api',methods=['GET'])
def addfaces_api():
    return "Welcome to Family Mem"

@app.route('/personalinfo', methods=['POST'])
def coll_personal_info():
    name = request.form.get("YourName")
    directory = "C:\images" + name.replace(" ", "-")
    patientname = request.form.get("PatientName")
    relationship = request.form.get("Relationship")

    try:
        os.mkdir(directory)
    except FileExistsError:
        print("The folder %s already exists! Overwriting." % directory)
    else:
        print("Succesfully created folder %s!" % directory)

    info = {"Name": name, "PatientName": patientname, "Relationship": relationship, "Directory": directory}

    show_window()
    count = 0
    input = input_method()
    start_time = time.time()

    # Specify the Classifier for the Cascade
    global face_cascade
    face_cascade = cv2.CascadeClassifier('./Cascades/data/haarcascade_frontalface_default.xml')

    # Start Capturing Video from Default Webcam
    webcam_capture = cv2.VideoCapture(0)

    # number of picture to be taken by the camera
    global num_pics
    num_pics = 3
    global font
    global font_size
    global white
    global cyan
    global red
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_size = 0.75
    white = (255, 255, 255)
    cyan = (255, 255, 0)
    red = (0, 0, 255)

    while (True):
        # Capture frame-by-frame
        if input == "screen":
            screen_pil = ImageGrab.grab(bbox=(0, 0, width, height))
            screen_np = np.array(screen_pil)
            frame = cv2.resize(cv2.cvtColor(screen_np, cv2.COLOR_BGR2RGB), (int(width / 1.5), int(height / 1.5)))
        else:
            ret, frame = webcam_capture.read()

        processed_frame, new_count = process_frame(frame, info, count, num_pics)
        count = new_count
        # processed_screen = cv2.resize(processed_screen_full, dsize=(800, 450), interpolation=cv2.INTER_LINEAR)

        # Display the resulting frame
        cv2.imshow(input.title(), processed_frame)

        # Stop if user presses 'q' or specified number of pictures have been taken
        if (cv2.waitKey(20) & 0xFF == ord('q')) or new_count >= num_pics:
            break

    # Stop getting webcam input and close all windows at the end of the program
    webcam_capture.release()
    cv2.destroyAllWindows()

    if count >= num_pics:
        print("Face Data Collection Complete")
        end_time = time.time()
        total_time = end_time - start_time
        print("Program complete in %f seconds" % total_time)
    else:
        print("Cancelled")
    return "pics uploaded"


if __name__ == "__main__":
    app.run(debug=True)

