import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import *
import matplotlib.pyplot as plt

num_classes = 10
input_shape = (28, 28, 1)

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = keras.Sequential()

# Convolution Layer 1
model.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1)))  # 32 different 3x3 kernels -- so 32 feature maps
model.add(BatchNormalization(axis=-1))  # normalize each feature map before activation
convLayer01 = Activation('relu')  # activation
model.add(convLayer01)

# Convolution Layer 2
model.add(Conv2D(32, (3, 3)))  # 32 different 3x3 kernels -- so 32 feature maps
model.add(BatchNormalization(axis=-1))  # normalize each feature map before activation
model.add(Activation('relu'))  # activation
convLayer02 = MaxPooling2D(pool_size=(2, 2))  # Pool the max values over a 2x2 kernel
model.add(convLayer02)

# Convolution Layer 3
model.add(Conv2D(64, (3, 3)))  # 64 different 3x3 kernels -- so 64 feature maps
model.add(BatchNormalization(axis=-1))  # normalize each feature map before activation
convLayer03 = Activation('relu')  # activation
model.add(convLayer03)

# Convolution Layer 4
model.add(Conv2D(64, (3, 3)))  # 64 different 3x3 kernels -- so 64 feature maps
model.add(BatchNormalization(axis=-1))  # normalize each feature map before activation
model.add(Activation('relu'))  # activation
convLayer04 = MaxPooling2D(pool_size=(2, 2))  # Pool the max values over a 2x2 kernel
model.add(convLayer04)
model.add(Flatten())  # Flatten final 4x4x64 output matrix into a 1024-length vector

# Fully Connected Layer 5
model.add(Dense(512))  # 512 FCN nodes
model.add(BatchNormalization())  # normalization
model.add(Activation('relu'))  # activation

# Fully Connected Layer 6
model.add(Dropout(0.2))  # 20% dropout of randomly selected nodes
model.add(Dense(10))  # final 10 FCN nodes
model.add(Activation('softmax'))  # softmax activation

model.summary()

batch_size = 128
epochs = 15

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

model.save("ocr_model_2")

score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
