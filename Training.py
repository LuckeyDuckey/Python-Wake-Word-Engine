import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
import os
import matplotlib.pyplot as plt

def train_model(features_path, labels_path, models_path, epochs=10, show_evaluation=False):

    #----- Data Prep -----

    features = np.load(features_path + "features.npy")
    labels = np.load(features_path + "labels.npy")

    x = np.array(features)
    y = np.array(labels)

    x = x.reshape(x.shape[0], x.shape[1], x.shape[2], 1)
    print(features.shape)

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1)#, random_state=42, shuffle=True)

    #----- Model Creation -----

    model = keras.Sequential()

    model.add(keras.layers.Conv2D(16, (3,3), 1, activation="relu", input_shape=(256,345,1)))
    model.add(keras.layers.MaxPool2D())
    model.add(keras.layers.Dropout(0.15))

    model.add(keras.layers.Conv2D(32, (3,3), 1, activation="relu"))
    model.add(keras.layers.MaxPool2D())
    model.add(keras.layers.Dropout(0.15))

    model.add(keras.layers.Conv2D(26, (3,3), 1, activation="relu"))
    model.add(keras.layers.MaxPool2D())
    model.add(keras.layers.Dropout(0.15))

    model.add(keras.layers.Flatten())

    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dropout(0.15))
    model.add(keras.layers.Dense(1, activation='sigmoid'))

    #----- Model Training -----

    model.summary()

    model.compile(
        loss=tf.losses.BinaryCrossentropy(),
        optimizer='adam',
        metrics=['accuracy']
    )


    hist = model.fit(X_train, y_train, epochs=epochs, verbose=2)#verbose=1
    model.save(models_path+"WWD.h5")

    fig = plt.figure()
    plt.plot(hist.history["loss"], color="teal", label="loss")
    #plt.plot(hist.history["val_loss"], color="orange", label="val_loss")
    fig.suptitle("Loss", fontsize=20)
    plt.legend(loc="upper left")
    plt.show()

    #----- Model Evalutanion -----

    pre = keras.metrics.Precision()
    re= keras.metrics.Recall()
    acc = keras.metrics.BinaryAccuracy()

    ypred = model.predict(X_test)
    pre.update_state(y_test, ypred)
    re.update_state(y_test, ypred)
    acc.update_state(y_test, ypred)

    print("Precision:"+str(pre.result().numpy())+", Recall:"+str(re.result().numpy())+", Accuracy:"+str(acc.result().numpy()))

    x = input("exit:")

path = os.path.dirname(os.path.realpath(__file__)) + "\\Data\\"
train_model(path, path, path, 10, True)
