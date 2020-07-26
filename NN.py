import csv
import numpy as np
import matplotlib.pyplot as plt
from ClassificationPlotter import plot_regions
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
from sklearn.model_selection import train_test_split

# read data 42000*785
# X1 = pd.read_csv('Text_vector_all.csv')
# X1  = X1.drop(columns = ["Unnamed: 0"])

# read test matrix
# y1 = pd.read_csv('ChannelVideo.csv')
# y1 = y1["like"] / (y1["dislike"] + y1["like"])

def NN(X1,y1):
    np.random.seed(1)
    X1_train, X1_val, y1_train, y1_val = train_test_split(X1, y1, test_size=0.2, random_state=1)
    print(X1_train.shape)
    print(X1_val.shape)
    print(y1_train.shape)
    print(y1_val.shape)

    np.random.seed(1)
    tf.random.set_seed(1)

    # model_1 = Sequential()
    # model_1.add(Dense(4, input_shape=(200,), activation='sigmoid'))
    # model_1.add(Dense(1, activation='sigmoid'))

    model = Sequential()
    model.add(Dense(32, activation='relu', input_dim=200))
    #model.add(Dropout(0.2))
    model.add(Dense(16, activation='relu'))
    # model.add(Dropout(0.2))
    model.add(Dense(1))

    opt = keras.optimizers.Adam(lr=0.01)
    model.compile(loss='mean_squared_error', optimizer="adam", metrics=['mse'])

    h = model.fit(X1_train, y1_train, batch_size=128, epochs=10, verbose=1, validation_data=(X1_val, y1_val))

    tra_score = model.evaluate(X1_train, y1_train, verbose=1)
    val_score = model.evaluate(X1_val, y1_val, verbose=1)

    print('Training Scores:  ', tra_score)
    print('Validation Scores:', val_score)

    plt.rcParams["figure.figsize"] = [8,4]
    plt.subplot(1,2,1)
    plt.plot(h.history['mse'], label='Training')
    plt.plot(h.history['val_mse'], label='Validation')
    plt.title('model mse')
    plt.ylabel('mse')
    plt.xlabel('epoch')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(h.history['loss'], label='Training')
    plt.plot(h.history['val_loss'], label='Validation')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend()
    plt.show()