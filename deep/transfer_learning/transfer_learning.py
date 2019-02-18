import matplotlib.pyplot as plt
from keras.layers import Dense,GlobalAveragePooling2D
from keras.applications import MobileNet
from keras.applications import MobileNetV2
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model



NUMERO_CLASSI = 7
EPOCHE = 5
TRAIN_PATH = './sport/training/'
VALID_PATH = './sport/validation/'


#Modello da cui partire:MobileNet a cui togliamo ultimo strato
base_model=MobileNet(weights='imagenet',include_top=False)

#Aggiungiamo livello al modello gia' addestrato
x=base_model.output
x=GlobalAveragePooling2D()(x)
#primo livello
x=Dense(1024,activation='relu')(x)
#secondo livello
x=Dense(1024,activation='relu')(x)
#terzo livello
x=Dense(512,activation='relu')(x)
#livello di output: softmax
preds=Dense(NUMERO_CLASSI,activation='softmax')(x)


# "fondo" il modello addestrato con quello che abbiamo creato
model=Model(inputs=base_model.input,outputs=preds)

model.summary()

#devono essere addestrati solo i nuovi livelli
for layer in base_model.layers:
    layer.trainable = False



#CREO TRAINING SET:
train_datagen = ImageDataGenerator()
                                   
train_generator=train_datagen.flow_from_directory(TRAIN_PATH, # this is where you specify the path to the main data folder
                                                  target_size=(224,224),
                                                  color_mode='rgb',
                                                  batch_size=32,
                                                  class_mode='categorical',
                                                  shuffle=True)

#stampa dizionario "etichetta:valore",per sapere l'indice delle classi
class_dictionary = train_generator.class_indices
print (class_dictionary)


#CREO VALIDATION SET

valid_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

valid_generator = valid_datagen.flow_from_directory(
                                                    directory=VALID_PATH,
                                                    target_size=(224, 224),
                                                    color_mode="rgb",
                                                    batch_size=32,
                                                    class_mode="categorical",
                                                    shuffle=True)





model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])


STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size

history = model.fit_generator(generator=train_generator,
                              steps_per_epoch=STEP_SIZE_TRAIN,
                              validation_data=valid_generator,
                              validation_steps=STEP_SIZE_VALID,
                              epochs=EPOCHE)



#Costruisco grafici per vedere accuratezza e loss sul training e validation set

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'b', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

#salvo il modello
model.save('modello.model')

