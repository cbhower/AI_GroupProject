from keras.callbacks import EarlyStopping
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam
from keras.layers.pooling import MaxPooling2D
from keras.utils import to_categorical
import numpy as np
import image_utils


X, Y = image_utils.load_data()

X_train = np.array(X[0:1200])
Y_train = np.array(Y[0:1200])

X_test = np.array(X[1200:])
Y_test = np.array(Y[1200:])

print(Y_train.shape)
print(Y_test.shape)


#(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

#print(Y_train[0])

num_epochs = 100

model = Sequential()
# Source specic input layer

source_layer = Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(224, 224, 3))
target_layer = Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(224, 224, 3))
source_layer.trainable = True
target_layer.trainable = True

model.add(source_layer)
model.add(target_layer)

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))


source_output_layer = Dense(3, activation='softmax')
target_output_layer = Dense(3, activation='softmax')
source_output_layer.trainable = False
target_output_layer.trainable = False

model.add(source_output_layer)
model.add(target_output_layer)

# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.0001, decay=1e-6),
              metrics=['accuracy'])

iteration = 0

while iteration < num_epochs:
    num_epochs -= 1

    if num_epochs % 2 == 0:
        source_layer.trainable = True
        source_output_layer.trainable = True
        # source specific layers on
        # x_train = source data
        # y_train = source labels
    else:
        target_layer.trainable = True
        target_output_layer.trainable = True

        # Target specific layers on
        # x_train = target data
        # y_train = target labels

    print(source_layer.get_weights())
    # train non-frozen layers
    model.fit(X_train, to_categorical(Y_train),
              batch_size=64,
              shuffle=True,
              epochs=1,
              validation_data=(X_test, to_categorical(Y_test)),
              callbacks=[EarlyStopping(min_delta=0.001, patience=3)])

    print(source_layer.get_weights())
    # Evaluate the model
    scores = model.evaluate(X_test, to_categorical(Y_test))

    print('Loss: %.3f' % scores[0])
    print('Accuracy: %.3f' % scores[1])

    source_layer.trainable = False
    target_layer.trainable = False
    source_output_layer.trainable = False
    target_output_layer.trainable = False

