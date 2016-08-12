#! /usr/bin/env python
# -*- coding: utf-8 -*-
import sys, os, os.path, caffe
import AsaUtilsMod as AsaUtils
import numpy as np
import skimage.io
import math

def makeDescriptor(fc6Feats,row,col):
    ro = (row - 128)/32 +1
    co = (col - 128)/32 +1

    #Completa
    descriptor = np.nanmax(fc6Feats, axis=0)

    #En cuatro
    fc6 = np.reshape(fc6Feats, (ro, co, 4096))
    div_r = math.floor(ro/2)
    div_c = math.floor(co/2)
    for i in range(2):
        for j in range(2):
            if (i+1)*div_r >= ro-1:
                a,b,c = fc6[i*div_r: ,j*div_c:(j+1)*div_c ,:].shape
                descriptor =  np.append(descriptor,np.nanmax(np.reshape(fc6[i*div_r: ,j*div_c:(j+1)*div_c ,:],(a*b, c)), axis=0))
            elif (j+1)*div_c >= co-1:
                a,b,c = fc6[i*div_r:(i+1)*div_r ,j*div_c: ,:].shape
                descriptor =  np.append(descriptor,np.nanmax(np.reshape(fc6[i*div_r:(i+1)*div_r ,j*div_c: ,:],(a*b, c)), axis=0))
            else:
                a,b,c = fc6[i*div_r:(i+1)*div_r ,j*div_c:(j+1)*div_c ,:].shape
                descriptor =  np.append(descriptor,np.nanmax(np.reshape(fc6[i*div_r:(i+1)*div_r ,j*div_c:(j+1)*div_c ,:],(a*b, c)), axis=0))

    #En 16
    fc6 = np.reshape(fc6Feats, (ro, co, 4096))
    div_r = math.floor(ro/4)
    div_c = math.floor(co/4)
    for i in range(4):
        for j in range(4):
            if (i+1)*div_r >= ro-1:
                a,b,c = fc6[i*div_r: ,j*div_c:(j+1)*div_c ,:].shape
                if a == 0 or b == 0:
                    descriptor = np.append(descriptor,fc6[i*div_r: ,j*div_c:(j+1)*div_c ,:])
                else:
                    descriptor =  np.append(descriptor,np.amax(np.reshape(fc6[i*div_r: ,j*div_c:(j+1)*div_c ,:],(a*b, c)), axis=0))
            elif (j+1)*div_c >= co-1:
                a,b,c = fc6[i*div_r:(i+1)*div_r ,j*div_c: ,:].shape
                if a == 0 or b == 0:
                    descriptor = np.append(descriptor,fc6[i*div_r:(i+1)*div_r ,j*div_c: ,:])
                else:
                    descriptor =  np.append(descriptor,np.amax(np.reshape(fc6[i*div_r:(i+1)*div_r ,j*div_c: ,:],(a*b, c)), axis=0))
            else:
                a,b,c = fc6[i*div_r:(i+1)*div_r ,j*div_c:(j+1)*div_c ,:].shape
                descriptor =  np.append(descriptor,np.amax(np.reshape(fc6[i*div_r:(i+1)*div_r ,j*div_c:(j+1)*div_c ,:],(a*b, c)), axis=0))
    print descriptor.shape
    return descriptor


MEAN_FILE = 'python/caffe/imagenet/ilsvrc_2012_mean.npy'
MODEL_FILE = 'models/bvlc_reference_caffenet/deploy.prototxt'
PRETRAINED = 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'

#IMAGE_NAMES_FILE = 'examples/images/MIT67/TrainImagesTitan.txt'
#IMAGE_NAMES_FILE = 'examples/images/MIT67/TestImagesTitan.txt'
#IMAGE_NAMES_FILE = 'examples/images/MIT67/titanSmall.txt'
IMAGE_NAMES_FILE = 'examples/images/miniMIT67/miniMIT67_list.txt'

OUT_DESCRIPTOR_FOLDER = 'data/Descriptors/'


LAYER = 'fc6'
BATCH_SIZE = 10 # for efficiency has to be same value than deploy.prototxt

NUM_FEATS = 4096
MIN_IMG_SIDE=256

STRIDE=32;
PATCH_ROW_SIZE=128
PATCH_COL_SIZE=128

#parse command line, get name of:i)file with imagenames and ii) name of path to store outputs 
inNames=AsaUtils.parseCmdLine(sys.argv[1:])

outType=0
keepClassFolders=0
imgNamesFile=IMAGE_NAMES_FILE
outputFolder=OUT_DESCRIPTOR_FOLDER

if inNames is not None :
    if inNames[0]!="":
        imgNamesFile = inNames[0]
        
    if inNames[1]!="":
        outputFolder = inNames[1]
 
    if inNames[2]!="":
        print inNames[2]
        outType=int(inNames[2])
        if outType!=1 and outType!=0:
            print 'Wrong flag use -p 0 to store patches feats or -p 1 to use maxPooling'
            sys.exit()

    if inNames[3]!="":
        keepClassFolders=int(inNames[3])
        if keepClassFolders!=1 and keepClassFolders!=0:
            print 'Wrong flag use -t 1 to keep class folder or -t 0 otherwise'
            sys.exit()


#set caffe 
caffe.set_phase_test()
caffe.set_mode_gpu()
net = caffe.Classifier(MODEL_FILE, PRETRAINED)

net.set_mean('data', np.load(MEAN_FILE))
net.set_raw_scale('data', 255)
net.set_channel_swap('data', (2,1,0))

imgFilenames=AsaUtils.getImgNames(imgNamesFile)

#loop to process all images in dataset
for i in range(0,len(imgFilenames)):
    print 'image ' + str(i+1) + ' of ' + str(len(imgFilenames))
    
    img = caffe.io.load_image(imgFilenames[i])
    
    #scale image keeping aspect ratio and MIN_IMG_SIDE on shorter side
    img = AsaUtils.ScaleImg(img,MIN_IMG_SIDE)
    #skimage.io.imsave('goodScale.jpg', img)    

    patchesList=AsaUtils.getImgPatches(img,PATCH_ROW_SIZE,\
                PATCH_COL_SIZE, STRIDE)

    nPatches=len(patchesList)
    
    print nPatches
    #for kk in range(0,nPatches):
    #    skimage.io.imsave(str('aux'+str(kk+1)+'.jpg'), patchesList[kk])    
    #loop for all patches selected from the image
    nBatches=int(nPatches/BATCH_SIZE)
    
    #get mem to store feats for image patches
    fc6Feats=np.zeros((nPatches, NUM_FEATS), dtype=np.float32)

    endBatch=0
    for j in range(nBatches):
        initBatch, endBatch=[j*BATCH_SIZE , (j+1)*BATCH_SIZE]
          
        #print 'batch of patches ' + str(initBatch+1) + ' to ' + \
        #str(endBatch) + ' of ' + str(nPatches)
        net.predict(patchesList[initBatch:endBatch])
        
        for k in range(len(net.blobs[LAYER].data)):
            fc6Feats[k+j*BATCH_SIZE,:]=net.blobs[LAYER].data[k].flatten()
    
    print fc6Feats.shape
    #if needed process last batch with remaining patches
    remainingPatches=nPatches%BATCH_SIZE
    if remainingPatches > 0:    
        #print 'batch of patches ' + str(nBatches*BATCH_SIZE+1) + ' to ' + \
        #str(nPatches) + ' of ' + str(nPatches)

        net.predict(patchesList[endBatch:nPatches])

        for k in range(remainingPatches):
            fc6Feats[k+nBatches*BATCH_SIZE,:]=net.blobs[LAYER].data[k].flatten()

    #store descriptors to file
    if outType==0:
        AsaUtils.savePatchFeats(fc6Feats,imgFilenames[i], outputFolder, keepClassFolders)
    else:
        #get final descriptor using max-pooling
        descriptor=np.nanmax(fc6Feats, axis=0)
        #descriptor = makeDescriptor(fc6Feats,img.shape[0],img.shape[1])
        #AsaUtils.saveDescriptor(descriptor,imgFilenames[i],str(i),'fc6', outputFolder,keepClassFolders)
        #print descriptor
        print len(descriptor)