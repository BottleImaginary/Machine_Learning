import os
import glob
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint

# Define network architecture
model = Sequential()
model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Set the paths and parameters
train_path = r'D:\WPy64-31090\scripts\BotolDet\myenv\Dataset\Dataset\train'
test_path = r'D:\WPy64-31090\scripts\BotolDet\myenv\Dataset\Dataset\test'
checkpoint_path = 'model_checkpoint.h5'

# Load the training data
train_file_list = glob.glob(os.path.join(train_path, '*.jpg'))
num_train_files = len(train_file_list)
X_train = []
y_train = []
for filename in train_file_list:
    image = cv2.imread(filename)
    if image is None:
        print('Read image failed:', filename)
        continue
    image = cv2.resize(image, (64, 64))
    X_train.append(image)
    label = 1 if 'botol' in filename else 0
    y_train.append(label)
X_train = np.array(X_train)
y_train = np.array(y_train)

# Load the test data
test_file_list = glob.glob(os.path.join(test_path, '*.jpg'))
num_test_files = len(test_file_list)
X_test = []
y_test = []
for filename in test_file_list:
    image = cv2.imread(filename)
    if image is None:
        print('Read image failed:', filename)
        continue
    image = cv2.resize(image, (64, 64))
    X_test.append(image)
    label = 1 if 'botol' in filename else 0
    y_test.append(label)
X_test = np.array(X_test)
y_test = np.array(y_test)

# Train the model
checkpoint_callback = ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, mode='min')
model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_test, y_test), callbacks=[checkpoint_callback])

# Save the trained model
model.save('bottle_detection_model.h5')