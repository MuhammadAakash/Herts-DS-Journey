import numpy as numpy
import pandas as pandas
from sklearn.cluster import KMeans
from sklearn import preprocessing
import cv2

# Load the image

def loadImage(imagePath):
    
    image = cv2.imread(imagePath)
    print("Image shape: ", image.shape())

    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # print("Image shape: ", image.shape)
    return image



loadImage("ratina.jpeg")