import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_hub as hub
import cv2
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)
classes =['happy','sad']
new_model=tf.keras.models.load_model('C:/Users/rudra/Downloads/mymood.h5')
facecas=cv2.CascadeClassifier("C:/Users/rudra/PycharmProjects/pythonProject/opencv/kuuu/face.xml")
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
while True:

    suc, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = facecas.detectMultiScale(gray, 1.1, 5)
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        facees = facecas.detectMultiScale(roi_gray)
        if len(facees) == 0:
            print("err error not found face")
        else:
            for (ex, ey, ew, eh) in facees:
                face_roi = roi_color[ey:ey + h, ex:ex + w]
    final = cv2.resize(face_roi, (224, 224))
    final = np.expand_dims(final, axis=0)
    final = final / 255.0
    if len(facees)!=0:
        p = new_model.predict(final)
        q = np.argmax(p)
        q=q%2
        print(q)
        cv2.putText(img, str(classes[q]), (100, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 222, 0), 3)
    cv2.imshow("imgg", img)

    if cv2.waitKey(1) & 0xFF == ord('x'):
        break