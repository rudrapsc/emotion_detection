import tensorflow as tf
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
data="C:/Users/rudra/Downloads/archive/train"
classes =['angry','disgust','fear','happy','neutral','sad','surprise']
for cat in classes:
    path=os.path.join(data,cat)
    for img in os.listdir(path):
        img_array=cv2.imread(os.path.join(path,img))
        plt.imshow(cv2.cvtColor(img_array,cv2.COLOR_BGR2RGB))
        plt.show()
        break
size=224
trainn=cv2.resize(img_array,(size,size))
plt.imshow(cv2.cvtColor(trainn,cv2.COLOR_BGR2RGB))
plt.show()
train_data = []


def create():
    a = 0
    for cat in classes:
        path = os.path.join(data, cat)
        classs = classes.index(cat)
        print(classs)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img))
                new = cv2.resize(img_array, (size, size))
                train_data.append([new, classs])
            except Exception as e:
                pass
    # print(a)
create()
# print(a)
import random
random.shuffle(train_data)
x=[]
y=[]
for f,l in train_data:
    x.append(f)
    y.append(l)

X=np.array(x).reshape(-1,size,size,3)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_hub as hub
import cv2
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

model=tf.keras.applications.MobileNetV2()
base_input=model.layers[0].input
base_output=model.layers[-1].output
final_output=layers.Dense(128)(base_output)
final_ouput=layers.Activation('relu')(final_output)
final_output=layers.Dense(64)(final_ouput)
final_ouput=layers.Activation('relu')(final_output)
final_output=layers.Dense(7,activation='softmax')(final_ouput)
new_model=keras.Model(inputs=base_input,outputs=final_output)
Y=np.array(y)
new_model.compile(
  optimizer='adam',
  loss='sparse_categorical_crossentropy',
  metrics=['accuracy'])
new_model.fit(X,Y,epochs=2)
model.save('C:/Users/rudra/Downloads/mymoo9d.h5')
