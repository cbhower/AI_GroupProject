# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 07:39:50 2019

@author: Christian
"""

from keras.layers import Input, Dense, concatenate
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras.datasets import cifar10, cifar100
from keras.layers.core import Dropout, Flatten
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam
from keras.layers.pooling import MaxPooling2D
from keras.utils import to_categorical

# Load Data
(X_train_source, Y_train_source), (X_test_source, Y_test_source) = cifar10.load_data()
(X_train_target, Y_train_target), (X_test_target, Y_test_target) = cifar100.load_data(label_mode = "coarse")


# Task specific inputs
source_inputs = Input(shape=(32,32,3), name = 'source_inputs')
target_inputs = Input(shape=(32,32,3), name = 'target_inputs')


# Target Layers
conv1 = Conv2D(32, kernel_size=(3, 3), activation='relu')(source_inputs)
conv2 = Conv2D(32, kernel_size=(3, 3), activation='relu')(target_inputs)

# Shared Layers
conc = concatenate([conv1, conv2])
pool1 = MaxPooling2D(pool_size=(2, 2))(conc)
drop1 = Dropout(0.25)(pool1)
flat1 = Flatten()(drop1)
dense1 = Dense(100, activation='relu')(flat1)

# Task Specific Output Layers
source_pred = Dense(10, activation='softmax', name = 'source_pred')(dense1)
target_pred = Dense(20, activation='softmax', name = 'target_pred')(dense1)


model = Model(inputs=[source_inputs, target_inputs], outputs= [source_pred, target_pred])
model.compile(loss='categorical_crossentropy',
              optimizer= Adam(lr=0.0001, decay=1e-6),
              metrics=['accuracy'])

model.fit([X_train_source, X_train_target], [to_categorical(Y_train_source), to_categorical(Y_train_target)],
      batch_size=100,
      shuffle=True,
      epochs=1,
      validation_data=([X_test_source, X_test_target], [to_categorical(Y_test_source), to_categorical(Y_test_target)]),
      callbacks=[EarlyStopping(min_delta=0.001, patience=3)])

model.summary()
