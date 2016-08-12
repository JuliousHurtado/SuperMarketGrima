import numpy as np
import scipy.io as sio
import os
import cv2
import time, random
from sklearn.externals import joblib
import matplotlib.pyplot as plt
from sklearn import svm, cross_validation
from sklearn.metrics import confusion_matrix
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

from readData import Data

class Exemplar(object):
    def __init__(self, HM = True, cal = True, peso_0 = 0.8, peso_1 = 0.01, linear = True):
        self.HM = HM
        self.cal = cal
        self.peso_0 = peso_0
        self.peso_1 = peso_1
        self.dict_class = {0: self.peso_0, 1: self.peso_1}
        self.linear = linear

    def getCls(self,elem,train_data,data):

        train = []
        label = []
        data_extra = []
        label_extra = []

        for dat in data:
            if elem['label'] == dat['label']:
                if elem['name'] != dat['name']:
                    data_extra.append(dat['desc'])
                    label_extra.append(1)
            else:
                data_extra.append(dat['desc'])
                label_extra.append(1)

        for dat in train_data:
            if elem['label'] == dat['label']:
                if elem['name'] != dat['name']:
                    train.append(dat['desc'])
                    label.append(1)
            else:
                train.append(dat['desc'])
                label.append(1)

        train.append(elem['desc'])
        label.append(0)

        self.dict_class = {0: self.peso_0, 1: self.peso_1}
        cont = True
        while cont:
            if self.linear:
                linear_svm = svm.LinearSVC(C=100, verbose=False, class_weight = self.dict_class)
            else:
                linear_svm = svm.SVC(C=2048, kernel='rbf', gamma=2, class_weight= self.dict_class)
            linear_svm.fit(train,label)
            if self.HM:
                cont,train,label = self.hardMining(elem,linear_svm,data_extra,label_extra,train,label)
                #if cont:
                #    self.dict_class = {0: self.peso_0, 1: self.peso_1, 2: 0.8}
            else:
                cont = False
        if self.cal:
            cls = self.calibration(linear_svm,train,label)
        else:
            cls = linear_svm
        return cls

    def hardMining(self,elem,linear_svm,val_set,val_lab,train,label):
        """
        https://www.reddit.com/r/computervision/comments/2ggc5l/what_is_hard_negative_mining_and_how_is_it/
        """
        data = []
        cont = False
        num = 0
        for num,desc in enumerate(val_set):
            scr = linear_svm.predict(desc.reshape(1,-1))
            if (scr == 0):
                #if (elem['label'] != val_lab[num]):
                train.append(desc)
                label.append(1)
                cont = True
                num += 1
        return cont,train,label

    def calibration(self,cls,train,label):
        """
        http://fastml.com/classifier-calibration-with-platts-scaling-and-isotonic-regression/
        http://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html#sklearn.calibration.CalibratedClassifierCV
        """
        cal_cls = CalibratedClassifierCV(cls, cv='prefit', method='sigmoid')
        cal_cls.fit(train,label)
        return cal_cls

    def getBestScore(self,elem,data):
        max_score = None # o sera min score?, confidence score for self.classes_[1] where >0 means this class would be predicted.
        label = None

        for cls in data:
            if cls['label'] != 1:
                if self.cal:
                    scr = cls['cls'].predict_proba(elem.reshape(1,-1))
                    if max_score is None or scr[0][0] > max_score:
                        max_score = scr[0][0]
                        label = cls['label']
                else:
                    scr = cls['cls'].decision_function(elem.reshape(1,-1))
                    if max_score is None or scr < max_score:
                        max_score = scr
                        label = cls['label']
        return label,max_score

    def getTopScore2(self,data,cls,cant = 5):
        mini_name = []
        mini_score = []
        maxi_name = []
        maxi_score = []

        for elem in cls:
            scr = elem['cls'].predict_proba(data.reshape(1,-1))
            if (len(maxi_score) < cant):
                maxi_score.append(scr[0][0])
                mini_score.append(scr[0][0])
                maxi_name.append(elem['real_name'])
                mini_name.append(elem['real_name'])
                maxi_score,maxi_name = (list(t) for t in zip(*sorted(zip(maxi_score, maxi_name), reverse=True)))
                mini_score,mini_name = (list(t) for t in zip(*sorted(zip(mini_score, mini_name))))
            else:
                if maxi_score[4] < scr[0][0]:
                    maxi_score[4] = scr[0][0]
                    maxi_name[4] = elem['real_name']
                    maxi_score,maxi_name = (list(t) for t in zip(*sorted(zip(maxi_score, maxi_name), reverse=True)))
                if mini_score[4] > scr[0][0]:
                    mini_score[4] = scr[0][0]
                    mini_name[4] = elem['real_name']
                    mini_score,mini_name = (list(t) for t in zip(*sorted(zip(mini_score, mini_name))))
        return maxi_name,mini_name,maxi_score,mini_score

    def getTopScore3(self,data,cls,cant = 5):
        name = []
        score = []
        label = []

        for elem in cls:
            scr = elem['cls'].predict_proba(data.reshape(1,-1))
            if (len(score) < cant):
                score.append(scr[0][0])
                name.append(elem['real_name'])
                label.append(elem['label'])
                score,name,label = (list(t) for t in zip(*sorted(zip(score, name, label), reverse=True)))
            else:
                if score[4] < scr[0][0]:
                    score[4] = scr[0][0]
                    name[4] = elem['real_name']
                    label[4] = elem['label']
                    score,name,label = (list(t) for t in zip(*sorted(zip(score, name, label), reverse=True)))
        return name,score,label
