# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 10:28:50 2019

@author: Christian
"""

from keras.layers import Input, Dense, concatenate, Activation
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras.datasets import cifar10, cifar100
from keras.layers.core import Dropout, Flatten
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam
from keras.layers.pooling import MaxPooling2D
from keras.utils import to_categorical
import numpy as np


(X_train_source, Y_train_source), (X_test_source, Y_test_source) = cifar10.load_data()

# source input model
source_inputs = Input(shape=(32,32,3))
conv1 = Conv2D(32, kernel_size=(3, 3), activation='relu')(source_inputs)
source_input_model = Model(inputs = source_inputs, outputs = conv1)
source_input_model.summary() 


# target input model
target_inputs = Input(shape=(32,32,3))
conv2 = Conv2D(32, kernel_size=(3, 3), activation='relu')(target_inputs)
target_input_model = Model(inputs = target_inputs, outputs = conv2)


# intermediate model
int_inputs = Input(shape=(30,30,32))
x = MaxPooling2D(pool_size=(2, 2))(int_inputs)
x = Dropout(0.25)(x)

x = Conv2D(64, kernel_size=(3, 3), activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Dropout(0.25)(x)

x = Conv2D(128, kernel_size=(3, 3), activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Dropout(0.25)(x)

x = Flatten()(x)
int_outputs = Dense(512, activation='relu')(x)

int_model = Model(inputs = int_inputs , outputs = int_outputs)
int_model.summary()

# source output model
source_pred_inputs = Input(shape=(512,)) 
source_pred = Dense(10, activation='softmax')(source_pred_inputs)
source_pred_model = Model(inputs=source_pred_inputs, outputs = source_pred)
source_pred_model.summary() 

# target output model
target_pred_inputs = Input(shape=(32,32,3))
target_pred = Dense(20, activation='softmax')(target_pred_inputs)
target_pred_model = Model(inputs=target_pred_inputs, outputs = target_pred)


# Overall model
model_inputs = Input(shape=(32,32,3))
m = source_input_model(model_inputs)
m = int_model(m)
model_pred = source_pred_model(m)

model = Model(inputs=model_inputs , outputs = model_pred)

# compile
model.compile(loss='categorical_crossentropy',
              optimizer= Adam(lr=0.0001, decay=1e-6),
              metrics=['accuracy'])

# fit model
model.fit(X_train_source, to_categorical(Y_train_source),
      batch_size=100,
      shuffle=True,
      epochs=1,
      validation_data=(X_test_source, to_categorical(Y_test_source)),
      callbacks=[EarlyStopping(min_delta=0.001, patience=3)])

model.summary()
