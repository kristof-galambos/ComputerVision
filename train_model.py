import numpy as np
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras import models, Sequential
from keras import layers
import cv2

def train_vgg(data_path, model_path, epochs):
    vgg = VGG16(include_top=False, pooling='avg', weights='imagenet', input_shape=(178, 218, 3))
    vgg.summary()
    # Freeze the layers except the last 2 layers
    for layer in vgg.layers[:-5]:
        layer.trainable = False
    # Check the trainable status of the individual layers
    for layer in vgg.layers:
        print(layer, layer.trainable)
    # Create the model
    model = models.Sequential()
    # Add the vgg convolutional base model
    model.add(vgg)
    # Add new layers
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(1, activation='sigmoid'))
    model.summary()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)
    train_generator = data_generator.flow_from_directory(
        data_path,
        target_size=(178, 218),
        batch_size=12,
        class_mode='binary')

    model.fit_generator(train_generator, epochs=epochs)
    model.save(model_path)
    print('model saved')


def train_dnn(data_path, model_path, epochs):
    model = Sequential()
    model.add(Dense(1024, input_shape=(2500,), activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.summary()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    data_generator = ImageDataGenerator(preprocessing_function=preprocess_for_dnn)
    train_generator = data_generator.flow_from_directory(
        data_path,
        target_size=(2500,),
        color_mode='grayscale',
        batch_size=12,
        class_mode='binary')

    model.fit_generator(train_generator, epochs=epochs)
    model.save(model_path)
    print('model saved')


def preprocess_for_dnn(image):
    # img = preprocess_input(image)
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # smaller = np.resize(cv2.resize(image, (50, 50)), (50, 50, 1))
    smaller = cv2.resize(image, (50, 50))
    return smaller.flatten()


def train_cnn(data_path, model_path, epochs):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(178, 218, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu'))
    # model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))

    model.summary()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)
    train_generator = data_generator.flow_from_directory(
        data_path,
        target_size=(178, 218),
        batch_size=12,
        class_mode='binary')

    model.fit_generator(train_generator, epochs=epochs)
    model.save(model_path)
    print('model saved')
