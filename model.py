import os
import csv
import math
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import keras
from keras.models import Sequential
from keras.models import Model
from keras.applications.inception_v3 import InceptionV3
from keras.applications.mobilenet import MobileNet
from keras.layers import Dense, Dropout, Flatten, Lambda
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import Cropping2D, Input, ZeroPadding2D
from keras import regularizers
from keras import backend as K


# Load csv files and read frame meta data
def load_csv(data_path="data/", csv_names=["long_driving_log.csv", "reverse_driving_log.csv", "swimming_driving_log.csv", "my_driving_log.csv"] ):
    data_samples = []
    for csv_name in csv_names:
        with open(data_path + csv_name) as csvfile:
            reader = csv.reader(csvfile)
            for line in reader:
                if abs(float(line[3])) > 0.1 or random.random() < 0.1: # Remove 90 % of straight-ish line driving
                    data_samples.append(line)
    return np.array(data_samples)

# Static dataset without generators
def build_dataset(data_samples, data_path="data/"):
    X = []
    y = []
    for sample in data_samples[1:]:
        center_img_path = data_path + sample[0]
        center_img = cv2.imread(center_img_path)
        steering_angle = float(sample[3])
        
        X.append(center_img)
        y.append(steering_angle)
    return np.array(X), np.array(y)

# Training / Validation batch generator
def generator(samples, batch_size=64, start_y=80):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = './data/IMG/' + batch_sample[0].split('/')[-1]
                if len(name.split(".")) != 3 or name.split(".")[-1] != "jpg":
                    continue # Not an image!
                center_image = cv2.imread(name)
                center_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)
                center_image_flipped = center_image[:, ::-1]
                steering_angle = float(batch_sample[3])

                images.append(center_image)
                angles.append(steering_angle)
                # Augment with flipped version:
                images.append(center_image_flipped)
                angles.append(-steering_angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)
       
# Two convolutional layers followed by two fully connected layers    
def simplish_conv_model(input_shape=(160, 320, 3), y_crop=(55, 25)):
    new_shape = (input_shape[0] - y_crop[0] - y_crop[1], input_shape[1], input_shape[2])
    model = Sequential()
    model.add(Cropping2D(cropping=(y_crop, (0,0)), input_shape=input_shape))
    
    model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=new_shape, output_shape=new_shape))
    model.add(Conv2D(64, (5, 5), padding="valid", activation="elu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((3, 3), (3, 3), 'valid'))
    
    model.add(Conv2D(256, (5, 5), padding="valid", activation="elu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((3, 3), (3, 3), 'valid'))
    model.add(Dropout(0.5))
    
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Dropout(0.6))
    
    model.add(Dense(42))
    model.add(Dense(1, use_bias=False)) # It should be equally likely to turn left and right => no bias.
    return model

# Hello Convolution World. This is the simplest model which was also able to be trained to complete one lap.
def simple_conv_model(input_shape=(160, 320, 3), y_crop=(55, 25)):
    new_shape = (input_shape[0] - y_crop[0] - y_crop[1], input_shape[1], input_shape[2])
    model = Sequential()
    # pre-processing layers:
    model.add(Cropping2D(cropping=(y_crop, (0,0)), input_shape=input_shape))
    model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=new_shape, output_shape=new_shape))
    
    # Convolutional layer
    model.add(Conv2D(16, (5, 5), padding="valid", activation="elu"))
    model.add(MaxPooling2D((4, 4), (4, 4), 'valid'))
    model.add(Dropout(0.5)) 
    
    # Fully connected -> steering angle output
    model.add(Flatten())
    model.add(Dense(1), use_bias=False)
    return model

# Transfer learning from https://keras.io/applications/#mobilenet
def mobile_net_model(input_shape=(160, 320, 3), y_crop=(50, 20)):
    # Define input layers with cropping and normalization:
    new_shape = (input_shape[0] - y_crop[0] - y_crop[1], input_shape[1], input_shape[2])
    input_tensor = Input(input_shape)
    # Crop image to remove car hood and background:
    cropping = Cropping2D(cropping=(y_crop, (0,0)), input_shape=input_shape)(input_tensor)
    # Normalize image to zero mean even variance to better condition model training:
    normalizing = Lambda(lambda x: x/255.0 - 0.5, input_shape=new_shape, output_shape=new_shape)(cropping)
    # Load MobileNet model trained on imagenet
    base_model = MobileNet(dropout=1e-3, include_top=False, weights=None, input_tensor=input_tensor)
    # Add output layer:
    flat = Flatten()(base_model.output)
    out = Dense(1, use_bias=False)(flat)
    model = Model(base_model.input, out)
 
    return model


# Transfer learning from https://keras.io/applications/#inceptionv3
def inception_v3_model(input_shape=(160, 320, 3), y_crop=(50, 20), layers_to_train=0):
    print(layers_to_train)
    # Define input layers with cropping and normalization:
    new_shape = (input_shape[0] - y_crop[0] - y_crop[1], input_shape[1], input_shape[2])
    input_tensor = Input(input_shape)
    # Crop image to remove car hood and background:
    cropping = Cropping2D(cropping=(y_crop, (0,0)), input_shape=input_shape)(input_tensor)
    # Normalize image to zero mean even variance to better condition model training:
    normalizing = Lambda(lambda x: x/255.0 - 0.5, input_shape=new_shape, output_shape=new_shape)(cropping)
    # Load InceptionV3 model trained on imagenet
    base_model = InceptionV3(include_top=False, weights='imagenet', input_tensor=input_tensor, input_shape=input_shape) 
    
    # Disable training of parameters for every InceptionV3 layer except the last layers_to_train:
    for layer in base_model.layers:
        layer.trainable = False
        
    for layer in base_model.layers[-1 : -(1 + layers_to_train) : -1]:
        print(layer)
        layer.trainable = True
        
    # Add output layer:
    flat = Flatten()(base_model.output)
    out = Dense(1, use_bias=False)(flat)
    
    model = Model(base_model.input, out)
    return model
    
def train_model(csv_data, model_initializer, batch_size=64, epochs=3, start_y=80):
    input_shape = (160, 320, 3)
    train_samples, validation_samples = train_test_split(csv_data, test_size=0.2)
    training_generator = generator(train_samples, batch_size=batch_size, start_y=start_y)
    validation_generator = generator(validation_samples, batch_size=batch_size, start_y=start_y)
    model = model_initializer(input_shape=input_shape)
    
    model.compile(loss='mse', optimizer='adam')
    history_object = model.fit_generator(training_generator,
                        steps_per_epoch=math.ceil(len(train_samples)/batch_size),
                        validation_data=validation_generator,
                        validation_steps=math.ceil(len(validation_samples)/batch_size),
                        epochs=epochs, verbose=1)
    return model, history_object
    
# Plot training and validation loss
def plot_history(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Mean Squared Error Loss')
    plt.ylabel('MSE')
    plt.xlabel('Epoch')
    plt.legend(['Training set', 'Validation set'], loc='upper right')
    plt.show()
 
#Analyze distribution of steering angles:
def analyze_dataset(data_path="data/"):
    csv_files = ["long_driving_log.csv", "reverse_driving_log.csv", "swimming_driving_log.csv", "my_driving_log.csv"] 
    #csv_files = ["driving_log.csv", "long_driving_log.csv", "reverse_driving_log.csv", "swimming_driving_log.csv", "my_driving_log.csv"] 
    y_all = []
    for csv_file in csv_files:
        csv_data = load_csv(data_path=data_path, csv_names=[csv_file])
        y = []
        for sample in csv_data[1:]:
            steering_angle = float(sample[3])
            y.append(steering_angle)
            y.append(-steering_angle)
        y_all.append(y)
        
    plt.hist(y_all, bins=30)
    plt.legend(csv_files)
    plt.show()
 
# Load data and train a model
def load_and_train(model_initializer, data_path="data/"):
    csv_data = load_csv(data_path=data_path)
    model, history_object = train_model(csv_data, model_initializer, batch_size=64, epochs=5, start_y=80)
    model.save('model.h5')
    return model, history_object

# Load data, train a model and plot loss over the epochs
def load_train_and_plot(data_path="data/"):
    #model_to_use = lambda input_shape: inception_v3_model(input_shape=input_shape, layers_to_train=32)
    #model_to_use = mobile_net_model
    model_to_use = simple_conv_model
    model, history = load_and_train(model_to_use, data_path=data_path)
    plot_history(history)
  
if __name__ == "__main__":  
    print("Welcome!")
    #load_train_and_plot()
    analyze_dataset()
    print("Goodbye!")
