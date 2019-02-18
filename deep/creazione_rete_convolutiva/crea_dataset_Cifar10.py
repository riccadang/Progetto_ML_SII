import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm

#DataSet CIFAR10 contenente un training set con 10 classi e 5000 esempi per classe ed
#un test set con 1000 esempi per classe. Verrano utilizzate solamente 4 classi

#CREAZIONE TEST SET
#METTERE IN DATADIR LA CARTELLA CONTENENTE IL DATASET:
DATADIR_TRAIN = "./train"
DATADIR_TEST = "./test"
#aggiungi le categorie che ti servono
CATEGORIES = ["airplane", "automobile","dog","horse"]
IMG_SIZE = 32

#creo training set
training_data = []
test_data = []

#Visualizzo un'immagine
for category in CATEGORIES:
    path = os.path.join(DATADIR_TRAIN, category)
    print (path)
    for img in os.listdir(path):
        # in grigio, ignoriamo il colore:
        img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
        plt.imshow(img_array, cmap='gray')
        plt.show()
        break
    break

#funzione per create dataset
def create_dataset(cartella,data):
    for category in CATEGORIES:
        path = os.path.join(cartella,category)
        class_num = CATEGORIES.index(category)
        print("\nClasse: " + category + " " + " -> Index: " + str(class_num))
        for img in tqdm(os.listdir(path)):  #itero su ogni foto
            try:
                img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # converto in array
                data.append([img_array, class_num])  # aggiungo al training/test set
            except Exception as e:
                pass

print ("****************  CREO TRAINING SET ***************")
create_dataset(DATADIR_TRAIN,training_data)
print ("****************  CREO TEST SET ***************")
create_dataset(DATADIR_TEST,test_data)

print("\n"+str(len(training_data))+" training samples")
print("\n"+ str(len(test_data))+" test samples")



#mischia i dati i modo random
import random
random.shuffle(training_data)
random.shuffle(test_data)

x_train = []
y_train = []

x_test = []
y_test = []

# separo features e label
for features,label in training_data:
    x_train.append(features)
    y_train.append(label)

for features,label in test_data:
    x_test.append(features)
    y_test.append(label)

#Salva i dati in modo da non doverli calcolare ogni volta che va usato il modello
import pickle

#Salva training set
pickle_out = open("x_train.pickle","wb")
pickle.dump(x_train, pickle_out)
pickle_out.close()

pickle_out = open("y_train.pickle","wb")
pickle.dump(y_train, pickle_out)
pickle_out.close()

#Salva test set
pickle_out = open("x_test.pickle","wb")
pickle.dump(x_test, pickle_out)
pickle_out.close()

pickle_out = open("y_test.pickle","wb")
pickle.dump(y_test, pickle_out)
pickle_out.close()

