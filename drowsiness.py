import cv2
from mtcnn.mtcnn import MTCNN
import pygame
from pygame import mixer
import os
from keras.models import load_model
import numpy as np
import smtplib
from email.mime.text import MIMEText

mixer.init()
#load the music file
pygame.mixer.music.load('audio.mp3')
cap=cv2.VideoCapture(0)
#load the cnn model for drowsiness detection
model = load_model('cnncat2.h5')
path = os.getcwd()
# assigning MTCNN() function to the variable detector
detector=MTCNN()
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
count=0
thicc=2
rpred=[99]
lpred=[99]
p=-1
#l is the percept sequence upto 100 instances stores the condition of the eye
l=[1 for _ in range(100)]
countc,counto=0,0
while True:
    ret,img=cap.read()
    #to get the height and width of the window frame
    height1, width1 = img.shape[:2]
    #for detecting the face in a particular frame
    faces = detector.detect_faces(img)
    # for putting a rectangle over the detected face
    for face in faces:
        x, y, width, height = face['box']
        x2, y2 = x + width, y + height
        cv2.rectangle(img, (x, y), (x2, y2), (255, 0, 0), 4)
        #break statement so as to break as soon as we detect one face
        break
    #for detecting left and right eye using haar cascade
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')
    reye = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_righteye_2splits.xml')
    leye = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_lefteye_2splits.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    right_eye = reye.detectMultiScale(gray)
    left_eye = leye.detectMultiScale(gray)
    #if face is detected
    if(faces!=[]):
     #o represents the current status of eye
     o=1
     #detecting the right eye using haar cascade
     for (x, y, w, h) in right_eye:
         r_eye = img[y:y + h, x:x + w]
         count=count+1
         r_eye = cv2.cvtColor(r_eye, cv2.COLOR_BGR2GRAY)
         r_eye = cv2.resize(r_eye, (24, 24))
         r_eye = r_eye / 255
         r_eye = r_eye.reshape(24, 24, -1)
         r_eye = np.expand_dims(r_eye, axis=0)
         #predicting whether the eye is closed or not
         predict_x = model.predict(r_eye)
         rpred = np.argmax(predict_x, axis=1)
         if (rpred[0] == 1):
             lbl = 'Open'
         if (rpred[0] == 0):
             lbl = 'Closed'
         break
     # detecting the left eye using haar cascade
     for (x, y, w, h) in left_eye:
        l_eye = img[y:y + h, x:x + w]
        count=count+1
        l_eye = cv2.cvtColor(l_eye, cv2.COLOR_BGR2GRAY)
        l_eye = cv2.resize(l_eye, (24, 24))
        l_eye = l_eye / 255
        l_eye = l_eye.reshape(24, 24, -1)
        l_eye = np.expand_dims(l_eye, axis=0)
        #predicting whether the eye is closed or not
        predict_x = model.predict(l_eye)
        lpred = np.argmax(predict_x, axis=1)
        if (lpred[0] == 1):
            lbl = 'Open'
        if (lpred[0] == 0):
            lbl = 'Closed'
        break
     #if both the eyes are closed then we can assign o as 0
     if (rpred[0] == 0 and lpred[0] == 0):
            o=0
            #display "closed" text on the screen
            cv2.putText(img, "Closed", (10, height - 20), font, 1, (255, 0, 0), 1, cv2.LINE_AA)
            #countc increases by 2 units as the person continues to close his eye
            countc+=2
            counto=0
            # if(rpred[0]==1 or lpred[0]==1):
     else:
            cv2.putText(img, "Open", (10, height - 20), font, 1, (255, 0, 0), 1, cv2.LINE_AA)
            countc=0
            #counto increases by 3 units if the person continues to open his eye
            counto+=3
     #replace the initial isntances with the current new instances of the eye
     for i in range(countc):
      l.remove(l[0])
      l+=[o]
     for i in range(counto):
         l.remove(l[0])
         l+=[o]
     #finding the average of the percept sequence
     p=sum(l)/len(l)
     #printing the status of the driver according to the percentage
     if(p>0.85):
         cv2.putText(img, "Status : Awake", (200, 30), font, 1, (255, 0, 0), 1, cv2.LINE_AA)
     elif p>=0.65:
         cv2.putText(img, "Status : Drowsy", (200, 30), font, 1, (255, 0, 0), 1, cv2.LINE_AA)
     else:
         cv2.putText(img, "Status : Sleeping", (200, 30), font, 1, (255, 0, 0), 1, cv2.LINE_AA)
     cv2.putText(img, 'Percentage:' + str(round((p)*100,2))+" %.", (100, height - 20), font, 1, (255, 0, 0), 1, cv2.LINE_AA)
     #if p is less the 0.65 that is the threshold value we play the alarm
     if(p<0.65):
             try:
                 #if the speaker is not busy the alarm will sound
                 if (pygame.mixer.music.get_busy() == False):
                     pygame.mixer.music.play(0)
             except:  # isplaying = False
                 pass
             #this is for thinckness of the border
             if (thicc < 16):
                 thicc = thicc + 2
             else:
                 thicc = thicc - 2
                 if (thicc < 2):
                     thicc = 2
             #to display the red alert border on the screen
             cv2.rectangle(img, (0, 0), (width1, height1), (0, 0, 255), thicc)
    else:
        #if face is not detected then display FACE NOT DETECTED for the driver to position his head properly
        cv2.putText(img, "FACE NOT DETECTED!!", (80, 250), font, 2, (0,255, 0), 3, cv2.LINE_AA)
    #showing the frame using imshow
    cv2.imshow('Video', img)
    k = cv2.waitKey(40) & 0xff
    if k == 27:
        break
# Sender and receiver email addresses
sender = 'gn800649@gmail.com'
receiver = 'pradeep.sudakar07@gmail.com'

# Email content
subject = 'Drowsiness Alert!'
body = 'The driver is feeling drowsy. Please take necessary action.'

# Compose the email message
msg = MIMEText(body)
msg['Subject'] = subject
msg['From'] = sender
msg['To'] = receiver

# Send the email using SMTP
smtp_server = 'smtp.gmail.com'
smtp_port = 587
smtp_username = 'pradeep.sudakar07@gmail.com'
smtp_password = 'Ialak123ialak'
smtp_conn = smtplib.SMTP(smtp_server, smtp_port)
smtp_conn.starttls()
smtp_conn.login(smtp_username, smtp_password)
smtp_conn.sendmail(sender, receiver, msg.as_string())
smtp_conn.quit()
if (p < 0.3):
    cv2.putText(img, "Drowsy", (10, height - 50), font, 1, (255, 0, 0), 1, cv2.LINE_AA)
    # Send email alert
    smtp_server = 'smtp.gmail.com'
    smtp_port = 587
    smtp_username = 'user'
    smtp_password = '1Q2W3E4R5T6Y7U8I9O0P'
    sender = 'gn800649@gmail.com'
    receiver = 'pradeep.sudakar07@gmail.com'
    subject = 'Drowsiness Alert!'
    body = 'The driver is feeling drowsy. Please take necessary action.'
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = sender
    msg['To'] = receiver
    smtp_conn = smtplib.SMTP(smtp_server, smtp_port)
    smtp_conn.starttls()
    smtp_conn.login(smtp_username, smtp_password)
    smtp_conn.sendmail(sender, receiver, msg.as_string())
    smtp_conn.quit()
 
cap.release()
cv2.destroyAllWindows()