import numpy as np
import scipy.io as sio
import os
import cv2
import time, random
from sklearn.externals import joblib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import Normalizer

from hashClass import KSH
from exemplar import Exemplar
from readData import Data
from test import Test

class Detector(object):
    def __init__(self):
        self.cant_iter = 1
        self.cant_tablas = 1
        self.cant_subconjunto = 0
        self.train_red = True
        self.m =  100 #500
        self.r = 3
        self.trn = 200 #2000
        self.train = True
        self.HM = True
        self.cal = True
        self.peso_0 = 1.0
        self.peso_1 = 0.0001
        self.dict_class = {0: self.peso_0, 1: self.peso_1}
        self.accu = True
        self.top = 1
        self.matrix = False
        self.display = True
        self.hashing = True
        self.modelo = "/basic/"
        self.guardar_modelos = False
        self.min_score = 0.0
        self.norma = False
        self.pca = True
        self.pca_com = 3000
        self.datos = 'SupermarketGRIMA'
        s = """
                    Parametros Generales:
                        Datos a utilizar: """ + str(self.datos) + """
                        cantidad iteraciones: """ + str(self.cant_iter) + """  
                        cantidad de Tablas: """ + str(self.cant_tablas) + """
                        cantidad subconjunto: """ + str(self.cant_subconjunto) + """
                        Normalizacion: """ + str(self.norma) + """
                        PCA: """ + str(self.pca) + """
                        PCA component: """ + str(self.pca_com) + """
                    Parametros Hash: 
                        Valor M: """ + str(self.m) + """
                        Valor r: """ + str(self.r) + """
                        Valor trn: """ + str(self.trn) + """
                    Parametros SVM: 
                        Se entrena nuevos clasificadores: """ + str(self.train) + """
                        Hard-Mining: """ + str(self.HM) + """
                        Calibracion: """ + str(self.cal) + """
                        Peso clase 0: """ + str(self.peso_0) + """
                        Peso clase 1: """ + str(self.peso_1) + """
                        Modelos utilizados: """ + self.modelo + """
                        Guardar modelos: """ + str(self.guardar_modelos) + """
                    Parametros Resultados:
                        Mostrar Accuracy: """ + str(self.accu) + """
                        Comprarar los Top: """ + str(self.top) + """
                        Mostrar matriz Confusion: """ + str(self.matrix) + """
                        Mostrar Imagenes top clasificadas: """ + str(self.display) + """
                        Utilizar Hash para resultados: """ + str(self.hashing) + """
                        Score minimo requerido: """ + str(self.min_score) + """
                    """
        print s
        self.start()
    
    def start(self):
        self.readData()
        #print np.unique(self.test_label)
        #self.resumenDatos()
        self.createClassifier()
        self.test()

    def procesamientoDatos(self,X_r,Y_r):
        if self.norma:
            norm = Normalizer()
            X_r = norm.fit(X_r).transform(X_r)
            Y_r = norm.transform(Y_r)

        if self.pca:
            #for i in range(2000,4000,100):
            #    pca = PCA(n_components=i)
            #    pca.fit(X_r)
            #    print "Nuevo:", i
            #    print pca.noise_variance_
            pca = PCA(n_components=self.pca_com)
            X_r = pca.fit(X_r).transform(X_r)
            Y_r = pca.transform(Y_r)

        return X_r,Y_r

    def readData(self):
        start_time = time.clock()

        read_data = Data(db = self.datos, train_red = self.train_red)
        if self.cant_subconjunto > 0:
            self.train_data,self.train_label,self.train_name = read_data.subconjuntoDatos(True,self.cant_subconjunto)
            self.test_data,self.test_label,self.test_name = read_data.subconjuntoDatos(False,self.cant_subconjunto)
        else:
            self.train_data,self.train_label,self.train_name = read_data.getTrain()
            self.test_data,self.test_label,self.test_name = read_data.getTest()

        self.train_data,self.test_data = self.procesamientoDatos(self.train_data,self.test_data)
        print self.train_data.shape
        print self.test_data.shape
        print("Read Files --- %s seconds ---" % (time.clock() - start_time))

    def createSVM(self):
        self.exemplar = Exemplar( HM = self.HM, cal = self.cal, peso_0 = self.peso_0, peso_1 = self.peso_1)
        j = 1
        for i in range(len(self.ksh)):
            for key in self.ksh[i].table.keys():
                for elem in self.ksh[i].table[key]:
                    if (j%100) == 0:
                        print str(j) + " de  " + str(len(self.train_label))
                    j += 1
                    if self.train:
                        elem['cls'] = self.exemplar.getCls(elem,self.train_all_data,self.ksh[i].table[key])
                        if self.guardar_modelos:
                            joblib.dump(dict['cls'], 'modelos/' + self.datos + self.modelo + dict['name']) 
                    else:
                        elem['cls'] = joblib.load('modelos/' + self.datos + self.modelo + dict['name'])

    def createTableHash(self):
        self.ksh = []
        for j in range(self.cant_tablas):
            ksh_temp = KSH(self.train_data,self.train_label,self.train_all_data, m = self.m, r = self.r, trn = self.trn)

            while ksh_temp.verificarHash(self.test_data,self.test_label) < 0.9:
                ksh_temp = KSH(self.train_data,self.train_label,self.train_all_data, m = self.m, r = self.r, trn = self.trn)
            self.ksh.append(ksh_temp)
            ksh_temp.lenBucket()

    def createClassifier(self):
        start_time = time.clock()

        self.train_all_data = []
        for i in xrange(len(self.train_data)):
            dict = {}
            dict['desc'] = self.train_data[i]
            dict['label'] = self.train_label[i]
            dict['name'] = self.train_name[i].replace("/","_")
            dict['real_name'] = self.train_name[i]
            self.train_all_data.append(dict)
        print("Preparar datos --- %s seconds ---" % (time.clock() - start_time))

        start_time = time.clock()
        self.createTableHash()
        print("Create Tables --- %s seconds ---" % (time.clock() - start_time))

        start_time = time.clock()
        self.createSVM()
        print("Train Classifiers --- %s seconds ---" % (time.clock() - start_time))

    def resumenDatos(self):
        f = open('resumenDatos.txt','w')
        for i in range(1,121):
            f.write(self.index_class[i] + '\n')
            f.write(str(len(np.where(self.train_label == i)[0]))+ '\n')
            f.write(str(len(np.where(self.test_label == i)[0]))+ '\n')
            f.write('\n\n')

    def test(self):
        promedio = 0
        maxi = 0
        mini = 100
        test = Test(self.ksh,self.exemplar,self.min_score)
        for i in range(self.cant_iter):

            if self.accu:
                #acc = self.getAccuracyPerClass(self.test_data,self.test_label,self.train_all_data)
                acc = test.getAccuracy(self.test_data,self.test_label)
            if self.matrix:
                test.matrixConfusion(self.test_data,self.test_label,self.train_label,self.train_all_data)
            if self.display:
                test.getBestScoreImage(self.test_data,self.test_name,"./Resultados/mini_dataset.p")
                #test.visualizadorTopClsForTest(self.test_data,self.test_name,self.train_all_data)

if __name__ == "__main__":
    Detector()