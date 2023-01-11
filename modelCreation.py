import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from keras.layers import  Dense, Flatten
from keras.models import Model
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing.image import ImageDataGenerator
import keras
from keras.callbacks import ModelCheckpoint, EarlyStopping

ourBase = InceptionV3(input_shape=(224,224,3), include_top= False)

for layers in ourBase.layers:
    layers.trainable = False
x = Flatten()(ourBase.output)
x = Dense(units=2, activation='sigmoid')(x)

model = Model(ourBase.input, x)
model.compile(optimizer='adam', loss=keras.losses.binary_crossentropy, metrics=['accuracy'])
model.summary()

train_data = ImageDataGenerator(featurewise_center= True,
                                rotation_range=0.4,
                                width_shift_range=0.3,
                                horizontal_flip=True,
                                preprocessing_function=preprocess_input,
                                zoom_range=0.4, shear_range=0.4)

data_to_train = train_data.flow_from_directory(directory=r"C:\Users\Acer\Desktop\faultsdata",
                                               target_size=(224,224),
                                               batch_size=16)

print(data_to_train.class_indices)
mdcheck = ModelCheckpoint(filepath= ".keras_model.h5",
                          monitor="accuracy",
                          verbose=1,
                          save_best_only= True)

earlystp = EarlyStopping(monitor="accuracy",
                         min_delta= 0.1,
                         patience=5,
                         verbose=1)

callback = [mdcheck, earlystp]
history = model.fit(data_to_train, steps_per_epoch=5, epochs= 100, batch_size=16, callbacks=callback)
