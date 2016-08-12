import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import spatial
import math
import os

def extend(image):
    nrow, ncol, ncolor = image.shape
    n = int((nrow**2 + ncol**2)**.5//2 + 1)
    new = np.zeros((2*n, 2*n, ncolor))
    a = nrow//2
    b = ncol//2
    new[n-a:n-a+nrow, n-b:n-b+ncol, :] = image
    return new

def rotateImage(image, angle):
    image = extend(image)

    center = tuple(np.array(image.shape[0:2])/2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotate = cv2.warpAffine(image, matrix, image.shape[0:2], flags=cv2.INTER_LINEAR)

    return rotate

def whiteBackGroud(image):
    image = image.astype(np.uint8)
    nrow, ncol, ncolor = image.shape
    new = np.zeros((nrow, ncol, ncolor))
    #print image.shape

    img2gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(img2gray,(7, 7),0)

    for row in range(nrow):
        for col in range(ncol):
            if (blur[row,col] == 0):
                image[row,col] = [255,255,255]

    #cv2.imshow('Negativas',image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    return image

def pathVideo():
    list_video = []
    path = '/home/julio/Documents/Dataset/GroceryModels/videos/'
    for label in os.listdir(path):
        for video in os.listdir(path + label + '/'):
            if video[-4:] == '.avi':
                dict = {}
                dict['label'] = int(label)
                dict['path'] = path + label + '/' + video
                list_video.append(dict)
    return list_video

def main():
    list_video = pathVideo()
    skiped_frame = 10
    cont_img = 1
    for elem in list_video:
        cap = cv2.VideoCapture(elem['path']) 
        path = "/home/julio/Documents/Dataset/GroceryModels/img/" + str(elem['label']) + '/'
        if not os.path.exists(path):
            os.makedirs(path)
        contador = 1
        while cap.isOpened():
            if not cap.grab():
                break
            if contador == skiped_frame:
                contador = 0
                flag,frame = cap.retrieve()
                for angle in [0,15,30,45,60,75,90,105,120,135,150,175,180]:
                    img = rotateImage(frame,angle)
                    img = whiteBackGroud(img)
                    cv2.imwrite(path + str(cont_img) + '.png',img)
                    cont_img += 1

            contador += 1

    #nrow, ncol, ncolor = img.shape
    #for row in range(nrow):
    #    for col in range(ncol):
    #        if (img[row,col,0] == 0) and (img[row,col,1] == 0) and (img[row,col,2] == 0):
    #            img[row,col] = [255,255,255]

    #cv2.imshow('Negativas',new)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    cap.release()

if __name__ == "__main__":
    main()