import mediapipe as mp
import cv2
import math
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results

def draw_styled_landmarks(image, results):
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             ) 

def extract_keypoints(results):
    landmarks = results.pose_landmarks.landmark
    slx, sly = [landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER.value].y]
    srx, sry = [landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER.value].y]
    return slx, sly, srx, sry

'''def output(jarak):
    #jarak >80cm maju
    if 151 < jarak < 200:
        hasil= "maju"
        return hasil
    elif jarak < 150:
        hasil= "maju cepat"
        return hasil
    elif 219 > jarak > 200:
        hasil= "stop"
        return hasil
    elif jarak > 220:
        hasil= "mundur"
        return hasil'''

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

#jarak 1.0
#y = [60,80,100,120,150,170,190,210,230,250,270,290]
#x = [365,300,250,200,153,137,122,111,104,99,87,82]

#jarak 2.0
y = [60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240,250,260,270,280,290,300]
x = [413,367,315,281,255,235,202,194,176,162,152,141,135,125,122,117,112,105,100,96,93,89,87,85,80]
coff = np.polyfit(x,y,2) # y = Ax^2 + Bx + C

A, B, C = coff

arah = []
prediction = []
sequence = []
threshold = 0.7

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

                else:
                    arah.append(hasil2)

            if len(arah) > 5: 
                arah = arah[-5:]
            
            dcm = int(A*distance**2 + B*distance + C)
            dcmt = str(dcm)
            
            #cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, num_txt+' px' , (500,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(image, dcmt+' cm' , (3,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        
        # Show to screen
        cv2.imshow('Human Following', image)

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()