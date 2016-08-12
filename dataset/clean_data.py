import cv2
import numpy as np
from scipy import spatial
import os

def noMargin(img):
    height,width,c = img.shape
    new_img = []

    for x in range(height):
        if np.mean(img[x,:,:]) < 255.0:
            new_img.append(img[x,:,:])

    new_img = np.array(new_img)
    final_img = []
    for y in range(width):
        if np.mean(new_img[:,y,:]) < 255.0:
            final_img.append(new_img[:,y,:])

    return cv2.flip(np.array(final_img),2)

def addMargin(img):
    constant= cv2.copyMakeBorder(img,10,10,10,10,cv2.BORDER_CONSTANT,value=[255,255,255])
    return constant

def readData():
    path = '/home/julio/Documents/Dataset/GroceryModels/img/'
    path2 = '/home/julio/Documents/Dataset/GroceryModels/img_mod/'

    for label in os.listdir(path):
        for elem in os.listdir(path + label + '/'):
            #print path + label + '/' + elem
            img = cv2.imread(path + label + '/' + elem)
            img = noMargin(img)
            img = addMargin(img)
            #cv2.imshow('image',np.array(img))
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
            if not os.path.exists(path2 + label):
                os.makedirs(path2 + label)
            cv2.imwrite(path2 + label + '/' + elem,img)

if __name__ == "__main__":
    readData()