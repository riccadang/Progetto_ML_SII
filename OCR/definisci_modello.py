import tensorflow as tf
import numpy as np
import keras
print(tf.__version__)

print("==============================")
print("RETE NEURALE CON DATASET MNIST")
print("==============================")

EPOCHE = 4
NUMERO_CLASSI = 10


#28x28 immagini di numeri scritti a mano da 0-9
mnist = tf.keras.datasets.mnist

#x contiene immagine, y la predizione
(x_train,y_train),(x_test,y_test) = mnist.load_data()

#normalizzo i valori: da 0 a 255 -> 0,1
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

#Uso codifica onehot: abbiamo 10 classi: 0,1,2,3,4,5,6,7,8,9
y_train = keras.utils.to_categorical(y_train, NUMERO_CLASSI)
y_test = keras.utils.to_categorical(y_test, NUMERO_CLASSI)

print y_test[0]

#MODELLO:
model = tf.keras.models.Sequential();
#input layer:
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))


#hidden layer:
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))

model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))

#output layer:
model.add(tf.keras.layers.Dense(10,activation=tf.nn.softmax))

#parametri per training modello:
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#training:
model.fit(x_train, y_train, epochs=EPOCHE)




#valuto modello con il test set e lo salvo
val_loss, val_acc = model.evaluate(x_test, y_test)
print("Valutazione modello su test set:")

print("Loss: " + str(val_loss))
print("Accuratezza: "+str(val_acc))
model.save('modello.model')


