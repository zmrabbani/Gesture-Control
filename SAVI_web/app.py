from joblib import load
import random
from transformers import BertTokenizer
from transformers import TFBertForSequenceClassification
import tensorflow as tf
from keras.models import load_model

from werkzeug.utils import secure_filename
from PIL import Image
import os

import numpy as np
import time
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from transformers import BertTokenizer
from transformers import TFBertForSequenceClassification

from keras.models import load_model
import random
import serial

import speech_recognition as sr
import pyttsx3

from flask import *
import argparse
import cv2
import numpy as np
import os
import time
import mediapipe as mp
import serial
from keras.models import load_model
import serial
from logging import FileHandler,WARNING

from fungsi import *
from Waktu import *

import warnings
warnings.filterwarnings("ignore")

def Replace(value):
    value1 = float(value)
    value1 = round(value1,0)
    value1 = int(value1)
    value1 = str(value1)
    return value1
    
def speak(text):
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[1].id)
    engine.setProperty('rate', 175)
    engine.say(text)
    engine.runAndWait()

def chatbot ():
    #global chat
    chat = input("🧑 Kamu\t: ")
    return chat


#------------------------------------------------ FLASK ---------------------------------------------------------#
app   = Flask(__name__, static_url_path='/static')

camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 360)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

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

                    actions = np.array(['maju', 'mundur', 'berhenti'])

                    if len(sequence) == 30:
                        res = model.predict(np.expand_dims(sequence, axis=0))[0]
                        #resh= np.argmax(res)
                        resh = actions[np.argmax(res)]
                        #resh= request.form.get('resh')
                        predictions.append(np.argmax(res))
                        '''

                        if (resh=='maju'):
                            #arduino.write(str.encode('{"kontrol":"maju"}'))
                            hasil='maju'
                            print("wahana maju")
                            #time.sleep(1)
                            return hasil

                        elif (resh=='mundur'):
                            #arduino.write(str.encode('{"kontrol":"mundur"}'))
                            hasil='mundur'
                            print("wahana mundur")
                            #time.sleep(1)
                            return hasil

                        elif (resh=='berhenti'):
                            #arduino.write(str.encode('{"kontrol":"berhenti"}'))
                            hasil='berhenti'
                            print("wahana berhenti")
                            #time.sleep(1)
                            return hasil
                        '''
                        if np.unique(predictions[-10:])[0]==np.argmax(res): 
                            if res[np.argmax(res)] > threshold: 
                    
                                if len(sentence) > 0: 
                                    if actions[np.argmax(res)] != sentence[-1]:
                                        sentence.append(actions[np.argmax(res)])
                                        #print(sentence.append(actions[np.argmax(res)]))
                                        #print(sentence[-1])
                                        hasil=sentence[-1]
                                        #hasil= request.form.get(hasil)
                                        if hasil=='maju':
                                            print('maju')
                                            
                                            #for label in myroot.iter('label'):
                                                #label.text = str('maju')
                                            #mytree.write('templates/label.xml')
                                            #return hasil
                                        elif hasil=='mundur':
                                            print('mundur')
                                            #for label in myroot.iter('label'):
                                                #label.text = str('mundur')
                                            #mytree.write('templates/label.xml')
                                            #return hasil
                                        elif hasil=='berhenti':
                                            print('berhenti')
                                            #for label in myroot.iter('label'):
                                               #label.text = str('berhenti')
                                            #mytree.write('templates/label.xml')
                                            #return hasil
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

@app.route("/")
def home():
    return render_template("web.html")

@app.route('/video_feed', methods = ['GET', 'POST'])
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/audio", methods=['POST'])
def mic():
    global chat
    r = sr.Recognizer()
    
    with sr.Microphone(device_index=1) as source:
        r.adjust_for_ambient_noise(source)
        print("Berbicara")
        r.pause_threshold = 1
        audio = r.listen(source)
        
    try:
        print("Mengolah")   
        chat = r.recognize_google(audio, language ='id')
        print("Kamu Berbicara: " + chat)
        #chat = request.args.get('prediction_input')
        
    except Exception as e:
        print(e)
        print("Tidak dapat mendengar apapun")  
        return "None"
    
    return chat

def teks():
    global chat
    chat = request.args.get('prediction_input')
    return chat

# [Routing untuk API]		
@app.route("/get")
def apiDeteksi():
    chat = request.args.get('prediction_input')
    prechat = text_preprocessing_process(chat)
    input_text_tokenized = bert_tokenizer.encode(prechat,
                                             truncation=True,
                                             padding='max_length',
                                             return_tensors='tf')

    bert_predict = bert_load_model(input_text_tokenized)          # Lakukan prediksi
    bert_predict = tf.nn.softmax(bert_predict[0], axis=-1)         # Softmax function untuk mendapatkan hasil klasifikasi
    output = tf.argmax(bert_predict, axis=1)
    print(bert_predict)
    print(output)

    global response_tag
    response_tag = le.inverse_transform([output])[0]
    respons = random.choice(responses[response_tag])

    if(response_tag == 'SAVI.maju'):
        #arduino.write(str.encode('{"chatbot":"Maju"}'))
        speak(respons)
        return respons

    elif(response_tag == 'SAVI.mundur'):
        #arduino.write(str.encode('{"chatbot":"Mundur"}'))
        return respons

    elif(response_tag == 'SAVI.stop'):
        #arduino.write(str.encode('{"chatbot":"Stop"}'))
        return respons

    elif(response_tag == 'SAVI.slow'):
        #arduino.write(str.encode('{"chatbot":"lambat"}'))
        return respons

    elif(response_tag == 'SAVI.medium'):
        #arduino.write(str.encode('{"chatbot":"sedang"}'))
        return respons

    elif(response_tag == 'SAVI.fast'):
        #arduino.write(str.encode('{"chatbot":"cepat"}'))
        return respons
    
    elif(response_tag == 'SAVI.kanan'):
        #arduino.write(str.encode('{"mode":"hall", "direct":"right"}'))
        return respons

    elif(response_tag == 'SAVI.kiri'):
        #arduino.write(str.encode('{"mode":"hall", "direct":"left"}'))
        return respons

    elif(response_tag == 'SAVI.suhu'):
        #arduino.write(str.encode('{"chatbot":"temp"}'))
        #data = arduino.readline().decode("utf-8").strip('\n').strip('\r')
        data = Replace(data)
        return (respons + " " + data + " " + "derajat celcius")

    elif(response_tag == 'SAVI.hump'):
        #arduino.write(str.encode('{"chatbot":"hum"}'))
        #data = arduino.readline().decode("utf-8").strip('\n').strip('\r')
        data = Replace(data)
        return(respons + " " + data + " " + "RH")

    elif(response_tag == 'SAVI.jam'):
        return(respons + ' ' + get_time("%H %M") + ' ' + part)

    elif(response_tag == 'SAVI.hari'):
        return (respons + ' ' + get_time("%A"))

    elif(response_tag == 'SAVI.tanggal'):
        return(respons + ' ' + get_time("%d %B %Y"))
    
    elif(response_tag == 'SAVI.lokasi'):
        #arduino.write(str.encode('{"chatbot":"hum"}'))
        #data_gps = arduino.readline().decode("utf-8").strip('\n').strip('\r')
        return respons

    else:
        #speak(respons)
        return respons


@app.route("/tag")
def tag():
    return jsonify(response_tag=response_tag)

if __name__ == '__main__':
    listener = sr.Recognizer()
    player = pyttsx3.init()

    #arduino = serial.Serial('COM3',115200)

    #Pretrained Model
    PRE_TRAINED_MODEL = 'indobenchmark/indobert-base-p2'

    #Load tokenizer dari pretrained model
    bert_tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL)

    # Load hasil fine-tuning
    bert_load_model = TFBertForSequenceClassification.from_pretrained(PRE_TRAINED_MODEL, num_labels=41)
    bert_load_model.load_weights('BERTSavi98%_2023-07-29_19-53.h5')

    #speak(random.choice(responses[classes[11]]))
    parser = argparse.ArgumentParser(description="Flask app hand gesture control")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()

    #arduino = serial.Serial('COM3',115200)

    model = load_model('action_control.h5')
    mp_holistic = mp.solutions.holistic         # Holistic model
    mp_drawing = mp.solutions.drawing_utils     # Drawing utilities

    #Deploy di localhost
    app.run(host="0.0.0.0", port=args.port)
    #app.run(host="localhost", port=5000, debug=False)