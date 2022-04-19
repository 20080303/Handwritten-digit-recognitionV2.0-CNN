import pandas as pd
from keras import layers
from keras import models
from keras.datasets import mnist
from keras.utils.np_utils import to_categorical
#from keras.utils.multi_gpu_utils import multi_gpu_model
from keras          import datasets
import numpy as np
import matplotlib.pyplot as plt
#import os
#os.environ["CUDA_VISIBLE_DEVICES"]="0"  gpu暂时的罢工
train=pd.read_csv('publicgitcode\\mycode\\ai\\Handwritten-digit-recognitionV2.0 CNN\\train.csv')
Y_train = train["label"]                                    
X_train = train.drop(labels = ["label"],axis = 1)                 
del train                                                      
Y_train=Y_train.values.reshape(42000,1)                         
X_train =X_train.values.reshape(42000,28, 28, 1).astype('float32')   
X_train = X_train / 255.0  #归一化
Y_train = to_categorical(Y_train)
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), 
          activation='relu', 
          input_shape=(28, 28, 1))
          )
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dropout(0.25))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
model.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(X_train, Y_train, epochs=10, batch_size=64)
model.save('v1model.h5')

