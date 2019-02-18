# load_model_sample.py
from keras.models import load_model
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from keras.applications.mobilenet import preprocess_input


if __name__ == "__main__":
    
    # load model
    model = load_model("modello.model")
    CATEGORIE = ["RockClimbing", "badminton","croquet","polo","rowing","sailing","snowboarding"]

    
    print("ANALIZZO badg:")
    for filename in os.listdir('./sport/test/sailing/'):
        if filename.endswith(".jpg") or filename.endswith(".jpeg"):
            path = './sport/test/sailing/'+filename
           
            img_array = cv2.imread(path)
            
            
            new_array = cv2.resize(img_array,(224,224))
            processa = preprocess_input(new_array)
               
            pronta = processa.reshape(-1,224,224,3)
            
           

            pred = model.predict(pronta)
            
            predizione = np.argmax(pred)
            print(str(filename)+" "+ str(CATEGORIE[predizione]))






