from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras import models
from keras import layers

def training(data_path, model_path, epochs):
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