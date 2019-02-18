import tensorflow as tf
import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time
import pickle

IMG_SIZE = 32
NUM_CLASSI = 4
CATEGORIES = ["airplane", "automobile","dog","horse"]
#carico il dataset che ho creato tramite crea_dataset_Cifar10
pickle_in = open("x_train.pickle","rb")
x_train = np.array(pickle.load(pickle_in))
pickle_in = open("y_train.pickle","rb")
y_train = np.array(pickle.load(pickle_in))

pickle_in = open("x_test.pickle","rb")
x_test = np.array(pickle.load(pickle_in))
pickle_in = open("y_test.pickle","rb")
y_test = np.array(pickle.load(pickle_in))

print("x_train shape:", x_train.shape, "y_train shape:", y_train.shape)

# visualizzo un'immagine a caso
img_index = 7
label_index = y_train[img_index]
print ("y = " + str(label_index) + " " +(CATEGORIES[label_index]))
plt.imshow(x_train[img_index])
plt.show()

# Reshape input da (32, 32) a (32, 32, 1)
w, h = IMG_SIZE, IMG_SIZE
x_train = x_train.reshape(x_train.shape[0], w, h, 1)
x_test = x_test.reshape(x_test.shape[0], w, h, 1)

print("x_train shape:", x_train.shape, "y_train shape:", y_train.shape)
print(x_train.shape[0], 'train set')
print(x_test.shape[0], 'test set')

#Rappresentazione ONE-HOT
y_train = keras.utils.to_categorical(y_train, NUM_CLASSI)
y_test = keras.utils.to_categorical(y_test, NUM_CLASSI)

#Normalizzo i dati
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

model = tf.keras.Sequential()

# Il primo strato e un operatore di convoluzione con filtro 2x2
#  L'output consiste in 64 filtri.
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=2, strides=1, padding='same', activation='relu', input_shape=(IMG_SIZE,IMG_SIZE,1)))
#pooling
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2)))
#Dropout
model.add(tf.keras.layers.Dropout(0.2))

# Il secondo strato e un operatore di convoluzione con filtro 2x2
# L'output consiste in 32 filtri.
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=2, strides=1, padding='valid', activation='relu'))
#pooling
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2)))
#Dropout
model.add(tf.keras.layers.Dropout(0.2))


#L'output della rete convolutiva diventa input di una rete fully connected, con 2 layers.
# L'input alla sotto-rete corrisponde a vettori multidimensionali, perciò prima
# rendo l'input 1D.
model.add(tf.keras.layers.Flatten())
#primo strato fully-connected con 512 neuroni
model.add(tf.keras.layers.Dense(512, activation='relu'))
model.add(tf.keras.layers.Dropout(0.4))
#secondo strato fully-connected con 256 neuroni
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dropout(0.4))

# L'output della sottorete viene passato a una softmax per la classificazione.
model.add(tf.keras.layers.Dense(NUM_CLASSI, activation='softmax'))

#Ottimizzazione tramite la funzione adam, una variante di sgd
optimizer = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

#Utilizzo la funzione cross_entropy come loss function
model.compile(loss='categorical_crossentropy',
             optimizer=optimizer,
             metrics=['accuracy'])
model.summary()

start_time = time.time()
#Utilizzo un validation set pari al 20% del training set
history = model.fit(x_train,
         y_train,
         batch_size=64,
         epochs=10,
         validation_split=0.2)
#Salvo il modello addestrato
model.save('cifar_model.model')
#Calcolo tempo di addestramento
elapsed_time = time.time() - start_time
t = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
print('Elapsed time (learning):', t)
# Valutiamo il modello sul test set
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

#calcolo grafico valitadation e training
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()







