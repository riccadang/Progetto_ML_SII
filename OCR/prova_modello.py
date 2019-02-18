import tensorflow as tf
import os
a = os.getcwd()
new_model = tf.keras.models.load_model(a + '/modello.model')

mnist = tf.keras.datasets.mnist

#x contiene immagine, y la predizione
(x_train,y_train),(x_test,y_test) = mnist.load_data()

#normalizzo i valori: da 0 a 255 -> 0,1
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)
predictions = new_model.predict(x_test)
import numpy as np
import matplotlib.pyplot as plt

FERMATI = 10
for index,item in enumerate(x_test):

    if index == 10:
        break
    else:
        print("Valore predetto "+ str(np.argmax(predictions[index])) +", Valore corretto: " + str(y_test[index]))
        plt.imshow(x_test[index],cmap=plt.cm.binary)
        plt.show()


