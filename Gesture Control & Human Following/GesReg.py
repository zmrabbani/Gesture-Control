import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
import serial
from keras.models import load_model

#connect arduino mega
#arduino = serial.Serial('COM7',115200)
#time.sleep(2)

mp_holistic = mp.solutions.holistic         # Holistic model
mp_drawing = mp.solutions.drawing_utils     # Drawing utilities

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results

def draw_styled_landmarks(image, results):
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
    
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

colors = [(245,117,16), (117,245,16), (16,117,245), (16,117,245), (16,117,245), (16,117,245), (16,117,245), (16,117,245), (16,117,245), (16,117,245), (16,117,245)]
def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+num*30), (int(prob*100), 90+num*30), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
        
    return output_frame

# Path for exported data, numpy arrays
DATA_PATH = os.path.join('Dataset') 

# Actions that we try to detect
actions = np.array(['Maju', 'Mundur', 'Berhenti', 'Ikuti Saya', 'Kanan', 'Kiri', 'Mati', 'Putar Balik', 'Tambah Kec', 'Turunkan Kec', 'None'])

# Thirty videos worth of data
no_sequences = 90

# Videos are going to be 30 frames in length
sequence_length = 30

model = load_model('ges_control_20_90.h5')
#from tensorflow import keras
#model = keras.models.load_model('action_control.h5')

# 1. New detection variables (refined)
sequence = []
sentence = []
predictions = []
threshold = 0.7
 
sentence.append(' ')

cap = cv2.VideoCapture(0)
# Set mediapipe model 
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():

        # Read feed
        ret, frame = cap.read()

        # Make detections
        image, results = mediapipe_detection(frame, holistic)
        #print(results)
        
        # Draw landmarks
        draw_styled_landmarks(image, results)
        
        # 2. Prediction logic
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]
        
        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            resh= np.argmax(res)
            #print(actions[np.argmax(res)])
            predictions.append(np.argmax(res))
            '''
            if actions[np.argmax(res)]=='maju':
                #arduino.write(str.encode('{"kontrol":"maju"}'))
                print("wahana berhenti")
                time.sleep(1) 

            if actions[np.argmax(res)]=='mundur':
                #arduino.write(str.encode('{"kontrol":"mundur"}'))
                #print("wahana berhenti")
                time.sleep(1) 

            if actions[np.argmax(res)]=='berhenti':
                #arduino.write(str.encode('{"kontrol":"berhenti"}'))
                #print("wahana berhenti")
                time.sleep(1)
            '''
            
            
        #3. Viz logic
            if np.unique(predictions[-10:])[0]==np.argmax(res): 
                if res[np.argmax(res)] > threshold: 
                    
                    if len(sentence) > 0: 
                        if actions[np.argmax(res)] != sentence[-1]:
                            sentence.append(actions[np.argmax(res)])
                            global hasil
                            hasil = sentence[-1]
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

            if len(sentence) > 5: 
                sentence = sentence[-5:]

            # Viz probabilities
            image = prob_viz(res, actions, image, colors)

        hasil2= str(sentence[-1])  
        cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, hasil2, (280,30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Show to screen
        cv2.imshow('Gesture Control', image)

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()