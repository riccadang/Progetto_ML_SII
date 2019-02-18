import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
from keras.applications.mobilenet import preprocess_input




DATADIR = "./sport/test/"

#aggiungi le categorie che ti servono
CATEGORIES = ["RockClimbing", "badminton","croquet","polo","rowing","sailing","snowboarding"]

#controlliamo che la cartella sia corretta: visualizzo una sola foto
for category in CATEGORIES:
    path = os.path.join(DATADIR,category)
    
    for img in os.listdir(path):
        #in grigio, ignoriamo il colore:
        img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)
        plt.imshow(img_array, cmap='gray')
        plt.show()
        
        break
    break


#trasformo dimensione in 224x224 visto che non hanno dimensioni uguali
IMG_SIZE = 224
new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))


#creo data set
training_data = []

#funzione per create dataset
def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR,category)
        print (path)
        class_num = CATEGORIES.index(category)
        for img in tqdm(os.listdir(path)):  #itero su ogni foto
            try:
                
                img_array = cv2.imread(os.path.join(path,img))
                processata = preprocess_input(img_array)
                new_array = cv2.resize(processata, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass


create_training_data()

print(len(training_data))

#mischia i dati i modo random: non ho prima tutta una classe e poi le altre, mischio
import random
random.shuffle(training_data)

X = []
y = []

for features,label in training_data:
    X.append(features)
    y.append(label)



X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 3)

#Salva i dati in modo da non doverli calcolare ogni volta che va usato il modello
import pickle
pickle_out = open("X.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()

#PER LEGGERE I DATI CHE ABBIAMO SALVATO:
#pickle_in = open("X.pickle","rb")
#X = pickle.load(pickle_in)

#pickle_in = open("y.pickle","rb")
#y = pickle.load(pickle_in)
