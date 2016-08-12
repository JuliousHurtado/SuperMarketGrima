#! /usr/bin/env python
# -*- coding: utf-8 -*-
import sys, os, os.path, caffe
import numpy as np
import math
import scipy.io as sio 

def ScaleImg(img, minSide):
    inRow,inCol,dummy=np.shape(img)

    if inRow > inCol :
        dim = (int((float(minSide)*inRow)/float(inCol)),minSide)
    else:
        dim = (minSide,int((float(minSide)*inCol)/float(inRow)))
 
    return(caffe.io.resize_image(img, dim))

def getImgPatches(img, patchRowSize, patchColSize, stride):

    maxRow,maxCol,colorDepth=np.shape(img)

    initRow,endRow, initCol, endCol=[0,patchRowSize,0,patchColSize]

    #total number of images patches
    totalPatches=(1+int((maxRow-patchRowSize)/stride))*(1+int((maxCol-patchColSize)/stride));
    patchesArray=np.empty([totalPatches, patchRowSize, patchRowSize, colorDepth], dtype=np.float32);

    nPatch=0
    while True:
        if(initCol >= maxCol or endCol > maxCol) :
            initRow,endRow, initCol, endCol=[initRow+stride,endRow+stride,0,patchColSize]

        if(initRow >= maxRow or endRow > maxRow) :
            break

        patchesArray[nPatch]=img[initRow:endRow, initCol:endCol,0:colorDepth]
        nPatch+=1
        initCol, endCol=[initCol+stride, endCol+stride]
    return patchesArray

MEAN_FILE = '/home/julio/Documents/Models/ResNet/caffe/ResNet_mean.npy'
MODEL_FILE = '/home/julio/Documents/Models/ResNet/caffe/ResNet-152-deploy.prototxt'
PRETRAINED = '/home/julio/Documents/Models/ResNet/caffe/ResNet-152-model.caffemodel'
#MEAN_FILE = '/mnt/nas/GrimaRepo/jahurtado/models/resnet/ResNet_mean.binaryproto'
#MODEL_FILE = '/mnt/nas/GrimaRepo/jahurtado/models/resnet/ResNet-152-deploy.prototxt'
#PRETRAINED = '/mnt/nas/GrimaRepo/jahurtado/models/resnet/ResNet-152-model.caffemodel'

LAYER = 'fc1000'
BATCH_SIZE = 10 # for efficiency has to be same value than deploy.prototxt

NUM_FEATS = 1000
MIN_IMG_SIDE=224

STRIDE=32;
PATCH_ROW_SIZE=112
PATCH_COL_SIZE=112

if len(sys.argv) != 3:
    print("Formato: python getPatchResNet.py [INPUT FOLDER] [OUTPUT FOLDER] ")
    sys.exit(0)

IMAGE_INPUT_FOLDER = sys.argv[1]
OUT_DESCRIPTOR_FOLDER = sys.argv[2]


net = caffe.Net(MODEL_FILE, PRETRAINED, caffe.TEST)
#caffe.set_mode_gpu()
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_mean('data', np.load(MEAN_FILE).mean(1).mean(1))
transformer.set_transpose('data', (2,1,0))
transformer.set_raw_scale('data', 255)

net.blobs['data'].reshape(1,3,224,224)

imgFilenames = []
for elem in os.listdir(IMAGE_INPUT_FOLDER):
    if os.path.isfile(IMAGE_INPUT_FOLDER + elem):
        imgFilenames.append(IMAGE_INPUT_FOLDER + elem)

desc = []
names = []
k = 0
#loop to process all images in dataset
for i,name in enumerate(imgFilenames):
    print('image ' + str(i+1) + ' of ' + str(len(imgFilenames)))
    
    img = caffe.io.load_image(name)
    
    #scale image keeping aspect ratio and MIN_IMG_SIDE on shorter side
    img = ScaleImg(img,MIN_IMG_SIDE)   

    patchesList = getImgPatches(img,PATCH_ROW_SIZE,\
                PATCH_COL_SIZE, STRIDE)

    nPatches = len(patchesList)
    
    nBatches = int(nPatches/BATCH_SIZE)
    
    #get mem to store feats for image patches
    fc6Feats = np.zeros((nPatches, NUM_FEATS), dtype=np.float32)

    for j in range(nPatches):
        net.blobs['data'].data[...] = transformer.preprocess('data', patchesList[j])
        output = net.forward()
        fc6Feats[j] = output['prob']

    print(fc6Feats.shape)
    
    descriptor = np.nanmax(fc6Feats, axis=0)

    #filename2 = OUT_DESCRIPTOR_FOLDER + '/' + str(i)  + '.mat'
    #sio.savemat(filename2,  {'stored' : descriptor , 'file' : name})

    desc.append(descriptor)
    names.append(name)

    if len(desc) == 1000:
        filename2 = OUT_DESCRIPTOR_FOLDER + '/' + str(k)  + '.mat'
        sio.savemat(filename2,  {'stored' : desc , 'file' : names})
        desc = []
        names = []
        k += 1

filename2 = OUT_DESCRIPTOR_FOLDER + '/' + str(k)  + '.mat'
sio.savemat(filename2,  {'stored' : desc , 'file' : names})