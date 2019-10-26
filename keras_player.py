#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# customized feedforward neural network

# source 1: https://towardsdatascience.com/building-a-deep-learning-model-using-keras-1548ca149d37 (dense neural network 1)
# source 2: https://victorzhou.com/blog/keras-neural-network-tutorial/ (dense neural network 2)
# source 3: https://datascience.stackexchange.com/questions/28003/get-multiple-output-from-keras (multi input/output model 1)
# source 4: https://keras.io/getting-started/functional-api-guide/#multi-input-and-multi-output-models (multi input/output model 2)

# load all necessary libraries needed in this program
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from keras.models import Sequential
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
import tensorflow as tf
import pandas as pd
import numpy as np
import os

# supress tensorflow warning message
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
# supress tensorflow debug message
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 

# read in training data
train_df = pd.read_csv("data.csv")
# create a dataframe with all training data except the target column
train_X = train_df.drop(columns=["o_x", "o_y"])
# create a dataframe with only the target column
train_y_x = train_df[["o_x"]]
train_y_y = train_df[["o_y"]]
# get number of columns in training data
columns_num = train_X.shape[1]

# clean up the console
os.system("cls")

# define the input
input1 = Input(shape = (columns_num,))
# define the layers
x = Dense(25, activation = "relu")(input1)
x = Dense(5, activation = "relu")(x)
# define the outputs
output1 = Dense(1, activation = "linear")(x)
output2 = Dense(1, activation = "linear")(x)
# define the model
model = Model(inputs = input1, outputs = [output1, output2])
# compile the model using mse as a measure of model performance
model.compile(optimizer = "Adadelta", loss = "mean_squared_error", metrics = ["accuracy"])
# set early stopping to monitor if the model can't improve anymore
early_stopping_monitor = EarlyStopping(patience = 3)

# train model
print ("\nTraining the model with training data.\n")
model.fit(train_X, [train_y_x, train_y_y], validation_split = 0.2, epochs = 30, callbacks = [early_stopping_monitor])
# save the model as file to speed up the testing
model.save_weights("model.h5")

# test the model
print ("\nTesting the model with training data.\n")
predictions = model.predict(train_X[9:10])
predictions = np.around(predictions)
print (" = Real answer on [9:10] array input for the xy-axis is [2, 5]")
print (" = Predicted answer by the AI for the xy-axis is [%d, %d]" % (predictions[0], predictions[1]))