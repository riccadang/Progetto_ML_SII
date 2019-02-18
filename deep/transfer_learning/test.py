# load_model_sample.py
from keras.models import load_model
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from keras.applications.mobilenet import preprocess_input
import pickle
import keras

CLASSI = 7

if __name__ == "__main__":
    
    # load model
    model = load_model("modello.model")
    
    pickle_in = open("X.pickle","rb")
    X = pickle.load(pickle_in)
   
    pickle_in = open("y.pickle","rb")
    y = pickle.load(pickle_in)
    y = keras.utils.to_categorical(y,CLASSI )
    
    val_loss, val_acc = model.evaluate(X, y)
    print("================================")
    print("Valutazione modello su test set:")

    print("Loss: " + str(val_loss))
    print("Accuratezza: "+str(val_acc))





