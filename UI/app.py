from flask import *
import argparse
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
import serial
from keras.models import load_model
import serial
import xml.etree.ElementTree as ET
import math
from flask_mqtt import Mqtt
from logging import FileHandler,WARNING

#mytree = ET.parse('templates/label.xml')
#myroot = mytree.getroot()

app = Flask(__name__, template_folder="templates")

file_handler = FileHandler('errorlog.txt')
file_handler.setLevel(WARNING)

#model = load_model('action_control.h5')
#mp_holistic = mp.solutions.holistic         # Holistic model
#mp_drawing = mp.solutions.drawing_utils     # Drawing utilities

#arduino = serial.Serial('COM7',115200)
#time.sleep(2)

sequence = []
sentence = []
predictions = []
threshold = 0.7

y = [60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240,250,260,270,280,290,300]
x = [413,367,315,281,255,235,202,194,176,162,152,141,135,125,122,117,112,105,100,96,93,89,87,85,80]
coff = np.polyfit(x,y,2) # y = Ax^2 + Bx + C

A, B, C = coff

def output(jarak):
    #jarak 110-150cm maju
    if 162 < jarak < 235:
        hasil= "maju"
        return hasil
    #jarak 150cm keatas maju cepat
    elif jarak < 161:
        hasil= "maju cepat"
        return hasil
    #jarak 100-110cm kebawah berhenti
    elif 279 > jarak > 236:
        hasil= "stop"
        return hasil
    #jarak 90cm kebawah mundur
    elif jarak > 280:
        hasil= "mundur"
        return hasil

#URL='http://192.168.1.18'
#AWB= True
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
'''
def gen_frames():
    while True:
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
             with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
                while camera.isOpened():

                    # Read feed
                    ret, frame = cap.read()

                    # Make detections
                    image, results = mediapipe_detection(frame, holistic)
                    #print(results)
                    
                    # Draw landmarks
                    draw_styled_landmarks(image, results)
                    
                    if results.pose_landmarks:
                        lm = results.pose_landmarks.landmark

                        slx, sly = [lm[mp_holistic.PoseLandmark.LEFT_SHOULDER.value].x,lm[mp_holistic.PoseLandmark.LEFT_SHOULDER.value].y]
                        srx, sry = [lm[mp_holistic.PoseLandmark.RIGHT_SHOULDER.value].x,lm[mp_holistic.PoseLandmark.RIGHT_SHOULDER.value].y]
                        
                        h, w, c = frame.shape
                        
                        x1,y1= slx*w, sly*h
                        x2,y2= srx*w, sry*h

                        #print(x2)
                        #print(x1)
                        distance= int(math.sqrt((y2 - y1)**2+(x2 - x1)**2))
                        #print(distance)
                        num_txt = str(distance)
                        prediction.append(distance)

                        hasil2 = output(distance)
                        #print(hasil2)
                        

                        if np.unique(prediction[-1:])[0] == distance:
                            if len(arah) > 0:
                                if hasil2 != arah[-1]:
                                    arah.append(hasil2)
                                    print(arah[-1])
                                    hasil3=arah[-1]
                                    if hasil=='maju':
                                        print('maju')
                                        #arduino.write(str.encode('{"kontrol":"maju"}'))
                                        #time.sleep(1)
                                    if hasil=='mundur':
                                        print('mundur')
                                        #arduino.write(str.encode('{"kontrol":"maju"}'))
                                        #time.sleep(1)
                                    if hasil=='maju cepat':
                                        print('maju cepat')
                                        #arduino.write(str.encode('{"kontrol":"maju"}'))
                                        #time.sleep(1)
                                    if hasil=='stop':
                                        print('stop')
                                        #arduino.write(str.encode('{"kontrol":"maju"}'))
                                        #time.sleep(1)

                            else:
                                arah.append(hasil2)

                        if len(arah) > 5: 
                            arah = arah[-5:]
                        
                        dcm = int(A*distance**2 + B*distance + C)
                        dcmt = str(dcm)
                        
                        #cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
                        cv2.putText(image, num_txt+' px' , (500,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                        cv2.putText(image, dcmt+' cm' , (3,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    
                         
                    #ret, buffer = cv2.imencode('.jpg', image)
                    #frame = buffer.tobytes()

                    frame = cv2.imencode('.jpg', image)[1].tobytes()

                    yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result
                
                #hasil2= request.form.get(hasil)
                #return hasil
'''                

def gen_frames():
    while True:
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            sequence = []
            sentence = []
            predictions = []
            threshold = 0.7
            with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:  
                while camera.isOpened():
                    ret, frame = camera.read()

                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image.flags.writeable = False 
                    results = holistic.process(image)
                    image.flags.writeable = True                   
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                    # Draw face connections
                    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
                                            mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
                                            mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                                            ) 
                    # Draw pose connections
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                            mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                                            mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                            ) 
                    # Draw left hand connections
                    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                            mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                                            mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                            ) 
                    # Draw right hand connections  
                    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                                            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                            ) 

                    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
                    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
                    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
                    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
                    keypoints = np.concatenate([pose, face, lh, rh])
                    
                    sequence.append(keypoints)
                    sequence = sequence[-30:]

                    actions = np.array(['Maju', 'Mundur', 'Berhenti', 'Ikuti Saya', 'Kanan', 'Kiri', 'Mati', 'Putar Balik', 'Tambah Kec', 'Turunkan Kec', 'None'])

                    if len(sequence) == 30:
                        res = model.predict(np.expand_dims(sequence, axis=0))[0]
                        #resh= np.argmax(res)
                        resh = actions[np.argmax(res)]
                        #resh= request.form.get('resh')
                        predictions.append(np.argmax(res))
                        
                        if np.unique(predictions[-10:])[0]==np.argmax(res): 
                            if res[np.argmax(res)] > threshold: 
                    
                                if len(sentence) > 0: 
                                    if actions[np.argmax(res)] != sentence[-1]:
                                        sentence.append(actions[np.argmax(res)])
                                        #print(sentence.append(actions[np.argmax(res)]))
                                        #print(sentence[-1])
                                        hasil=sentence[-1]
                                        #hasil= request.form.get(hasil)
                                        if hasil=='Maju':
                                            print('maju')
                                            #arduino.write(str.encode('{"kontrol":"maju"}'))
                                            #time.sleep(1)
                                        elif hasil=='Mundur':
                                            print('mundur')
                                            #arduino.write(str.encode('{"kontrol":"mundur"}'))
                                            #time.sleep(1)
                                        elif hasil=='Berhenti':
                                            print('berhenti')
                                            #arduino.write(str.encode('{"kontrol":"berhenti"}'))
                                            #time.sleep(1)
                                        elif hasil=='Ikuti Saya':
                                            print('Ikuti Saya')
                                            #arduino.write(str.encode('{"kontrol":" "}')) #belum fix
                                            #time.sleep(1)
                                        elif hasil=='Kanan':
                                            print('Kanan')
                                            #arduino.write(str.encode('{"kontrol":"kanan"}'))
                                            #time.sleep(1)
                                        elif hasil=='Kiri':
                                            print('Kiri')
                                            #arduino.write(str.encode('{"kontrol":"kiri"}'))
                                            #time.sleep(1)
                                        elif hasil=='Mati':
                                            print('Mati')
                                            #arduino.write(str.encode('{"relay":"off"}'))
                                            #time.sleep(1)
                                        elif hasil=='Putar Balik':
                                            print('Putar Balik')
                                            #arduino.write(str.encode('{"kontrol":"putar balik"}'))
                                            #time.sleep(1)
                                        elif hasil=='Tambah Kec':
                                            print('Kec Sedang')
                                            #arduino.write(str.encode('{"kontrol":"Kecepatan_sedang"}'))
                                            #time.sleep(1)
                                        elif hasil=='Turunkan Kec':
                                            print('Kec Lambat')
                                            #arduino.write(str.encode('{"kontrol":"Kecepatan_lambat"}'))
                                            #time.sleep(1)   
                                else:
                                    sentence.append(actions[np.argmax(res)])
                                    #print(sentence.append(actions[np.argmax(res)]))

                        if len(sentence) > 5: 
                            sentence = sentence[-5:]
                            #if sentence[-1:]

                        colors = [(245,117,16), (117,245,16), (16,117,245)]
                        
                        for num, prob in enumerate(res):
                            cv2.rectangle(image, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
                            cv2.putText(image, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
                    
                        #return resh

                    cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
                    cv2.putText(image, ' '.join(sentence), (3,30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    
                    
                    
                    #ret, buffer = cv2.imencode('.jpg', image)
                    #frame = buffer.tobytes()

                    frame = cv2.imencode('.jpg', image)[1].tobytes()

                    yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result
    #hasil2= request.form.get(hasil)
    #return hasil


@app.route('/')
def input():
    return render_template('login.html')

@app.route('/UI', methods=['GET', 'POST'])
def display():
     if request.method == 'POST':  
        global result
        result = request.form
    
        return render_template("UI2.html", result=result)

@app.route('/video_feed', methods = ['GET', 'POST'])
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/requests', methods=['GET', 'POST'])
def tasks():
    global camera
    if request.method == 'POST':
        #result = request.form    
        if request.form.get('action1') == 'start':
            camera = cv2.VideoCapture(0)
        elif request.form.get('action2') == 'stop':
            camera.release()
            cv2.destroyAllWindows()
        else:
            pass 

    elif request.method == 'GET':
        return render_template('UI2.html', result=result) #, result=result
    
    return render_template('UI2.html', result=result)


if __name__== '__main__':
    parser = argparse.ArgumentParser(description="Flask app hand gesture control")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()

    arduino = serial.Serial('COM3',115200)
    time.sleep(2)

    model = load_model('action_control.h5')
    mp_holistic = mp.solutions.holistic         # Holistic model
    mp_drawing = mp.solutions.drawing_utils     # Drawing utilities

    app.run(host="0.0.0.0", port=args.port)