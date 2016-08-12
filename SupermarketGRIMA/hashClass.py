import theano.tensor as T
from theano import function

import scipy.io as sio
import os
import time, random
import numpy as np
from operator import itemgetter

import numpy.matlib
import scipy as sci
from numpy import linalg as LA

import sys

#Distance
import scipy.sparse
import scipy.spatial.distance

class DistanceMetric(object):
    #http://docs.scipy.org/doc/scipy/reference/spatial.distance.html

    def __init__(self, dist = 'man'):
        self.dist_met = dist

    def changeDistance(self, dist):
        self.dist_met = dist

    def distance(self, x, y, m = 2):
        if self.dist_met == 'man':
            return self.manhattan(x,y)
        if self.dist_met == 'eucl':
            return self.euclidian(x,y)
        if self.dist_met == 'cos':
            return self.cosine(x,y)
        if self.dist_met == 'dot':
            return self.dot(x,y)
        if self.dist_met == 'sign-alsh':
            return self.signALSH(x,y,m)

    def manhattan(self, x, y):
        """
        Computes the Manhattan distance between vectors x and y. Returns float.
        """
        if scipy.sparse.issparse(x):
            return numpy.sum(numpy.absolute((x-y).toarray().ravel()))
        else:
            return numpy.sum(numpy.absolute(x-y))

    def euclidian(self,x,y):
        return numpy.linalg.norm(x-y)

    def cosine(self,x,y):
        return scipy.spatial.distance.cosine(x,y)

    def dot(self,x,y):
        return np.inner(x,y)

    def signALSH(self,x,y,m):
        return np.inner(x,y)/np.sqrt(m/4 + x[-1])

class Evaluacion(object):
    def __init__(self, label_test):
        self.label_test = label_test
        self.label_dict = {}

        self.makeDictLabel()

    def makeDictLabel(self):
        for label in self.label_test:
            if label in self.label_dict.keys():
                self.label_dict[label] += 1
            else:
                self.label_dict[label] = 1

    def presicion(self,buenas,total):
        return float(buenas)/float(total)

    def recall(self,buenas,label):
        return float(buenas)/float(self.label_dict[label])

class SignLSH(object):
    def __init__(self, dimension_array, data, bits = 3):
        """
        dimension_array: Es la dimension del vector de caracteristicas que se va a utilziar
        bits: es el largo del vector binario
        """
        x = T.dvector('x')
        w = T.dmatrix('W')

        z = T.dot(w,x)
        self.f = function([x, w], z)

        self.dist = DistanceMetric(dist = 'dot')

        self.setW(bits,dimension_array)
        self.table = {}

        self.createHash(data)

    def setW(self, bits, dimension_array):
        self.m = np.random.uniform(-1,1,(bits,dimension_array))

    def getBinaryArray(self, desc):
        x_bin = self.f(desc,self.m)
        return ''.join([ '1' if i >= 0 else '0' for i in x_bin ])

    def createHash(self, data):
        for elem in data:
            x_bin = self.getBinaryArray(elem['desc'])

            if x_bin in self.table.keys():
                self.table[x_bin].append(elem)
            else:
                self.table[x_bin] = [elem]

    def nearNeighbors(self, elem_bin, elem = None, cant = 0):
        if elem_bin in self.table.keys(): 
            if cant == 0:
                return self.table[elem_bin]
            elif len(self.table[elem_bin]) <= cant:
                return self.table[elem_bin]
            else:
                dist_elem = []
                for elem_buck in self.table[elem_bin]:
                    dist = self.dist.distance(elem_buck['desc'],elem)
                    dist_elem.append((dist,elem_buck))
                dist_elem = sorted(dist_elem, key=itemgetter(0), reverse=True)
                #return dist_elem[:cant]
                return [ elem[1] for elem in dist_elem[:cant] ]
        else:
            #print "No element in buckets"
            return []

    def verificarHash(self,testdata,testgnd,cant = 0):
        buenos = 0
        malos = 0

        for i in range(len(testdata)):
            has = self.getBinaryArray(testdata[i])
            neighbors = self.nearNeighbors(has, testdata[i], cant)

            good = False
            for ind in neighbors:
                if ind['label'] == testgnd[i]:
                    good = True
                    break

            if good:
                buenos += 1
            else:
                malos += 1
        print "Precision de la tabla Hash: ",float(buenos)/float(buenos + malos)
        #print "Cantidad de buckets usados: ",len(self.table.keys())
        return float(buenos)/float(buenos + malos)

    def presRecall(self,testdata,testlabel,cant = 0):
        start_time = time.time()

        has = self.getBinaryArray(testdata)
        neighbors = self.nearNeighbors(has,testdata,cant)
        bucket = self.nearNeighbors(has)

        buenas = 0
        malas = 0
        total_buc = 0
        for nei in neighbors:
            if nei['label'] == testlabel:
                buenas += 1
            else:
                malas += 1
        time_total = time.time() - start_time
        for nei in bucket:
            if nei['label'] == testlabel:
                total_buc += 1

        if (buenas + malas) == 0:
            pres = 0
        else:
            pres = float(buenas)/float(buenas+malas)

        if total_buc == 0:
            rec = 0
        else:
            rec = float(buenas)/float(total_buc)

        return pres,rec,time_total

    def lenBucket(self):
        for key in self.table.keys():
            print "Elementos Bucket: ",len(self.table[key])

class KSH(object):
    def __init__(self, traindata,traingnd,data_total, m = 400, r = 7, trn = 2000):
        """
        dimension_array: Es la dimension del vector de caracteristicas que se va a utilizar
        bits: es el largo del vector binario
        """

        sample = random.sample(range(0,len(traindata)), m)
        label_index = random.sample(range(0,len(traindata)), trn)

        start_time = time.time()

        self.dist = DistanceMetric(dist = 'dot')

        self.anchor = np.copy(traindata[sample,:])
        KTrain = self.sqdist(traindata,self.anchor)

        S = self.pairwise_label_matrix(traingnd,label_index,r,trn)
        self.Y,self.Al = self.projection_optimization(KTrain,label_index,m,r,S,trn)

        start_time = time.time()
        self.table = {}
        self.dict_class = {}
        self.unique = max(np.unique(traingnd))
        self.createTable(self.Y,data_total,len(traindata))

    def createTable(self,Y,data_total,num_data):
        for i in range(num_data):
            bits = np.copy(Y[:,i])
            has = ''.join([ '1' if j >= 0 else '0' for j in bits ])

            if has in self.table.keys():
                self.table[has].append(data_total[i])
            else:
                self.table[has] = [data_total[i]]
                self.dict_class[has] = np.zeros(self.unique)
            self.dict_class[has][int(data_total[i]['label'])-1] = 1

    def sqdist(self,traindata,anchor):
        """
        Devuelve una matrix de distancia entre los datos y los anchor
        """

        KTrain = np.zeros((len(traindata),len(anchor)))

        data_aux = 0
        KTrain = self.distance(traindata,anchor)
        sigma = np.mean(KTrain)
        KTrain = np.exp(-KTrain/(2*sigma))
        mvec = np.mean(KTrain,axis = 0)
        KTrain = KTrain - np.tile(mvec.reshape(1,len(mvec)).T,len(traindata)).T
        return KTrain

    def distance(self,a,b):
        aa = np.sum(a,axis=1)**2
        bb = np.sum(b,axis=1)**2
        ab = a.dot(b.T)
        d = np.tile(aa.reshape(1,len(aa)).T, len(bb)) + np.tile(bb.reshape(len(bb),1),len(aa)).T - 2*ab
        return d

    def pairwise_label_matrix(self,traingnd,label_index,r,trn):
        """
        Return a matrix len(traingnd)xlen(traingnd): where i,j is r when label of i is same as label j
                                                     otherwise -r
        r is number of bits            
        """
        trngnd = np.copy(traingnd[label_index])

        temp = np.tile(trngnd.reshape(len(trngnd),1),trn).T \
                    - np.tile(trngnd.reshape(len(trngnd),1),trn)

        S_o = np.ones((trn,trn))*-1
        S_o[np.where( temp == 0 )] = 1

        return r*S_o

    def OptProjectionFast(self,K,S,a0,cn):
        n,m = K.shape
        cost = np.zeros((cn+2))
        y = K.dot(a0)
        y = 2*(1 + np.exp(-1*y))**-1-1
        cost[0] = (-1*y).T.dot(S).dot(y)
        cost[1] = np.copy(cost[0])

        a1 = np.copy(a0)
        delta = np.zeros((cn+2))
        delta[0] = 0
        delta[1] = 1
        beta = np.zeros((cn+1))
        beta[0] = 1

        for t in range(cn):
            alpha = (delta[t] - 1)/delta[t+1]
            v = a1 + alpha*(a1 - a0)
            y0 = K.dot(v)
            y1 = 2*(1 + np.exp(-1*y0))**-1-1
            gv = (-1*y1).T.dot(S).dot(y1)
            ty = (S.dot(y1))*(np.ones(n)-y1**2)
            dgv = (-1*K).T.dot(ty)

            flag = 0
            for j in range(50):
                b = (2**(j))*beta[t]
                z = v-dgv/b
                y0 = K.dot(z)
                y1 = 2*(1 + np.exp(-1*y0))**-1-1
                gz = (-1*y1).T.dot(S).dot(y1)
                dif = z - v
                gvz = gv + dgv.T.dot(dif) + b*dif.T.dot(dif)/2

                if gz <= gvz:
                    flag = 1
                    beta[t+1] = b
                    a0 = a1[:]
                    a1 = z[:]
                    cost[t+2] = gz
                    break

            if flag == 0:
                t = t-1
                break
            else:
                delta[t+2] = ( 1 + np.sqrt(1+4*delta[t+1]**2) )/2

            if abs(cost[t+2] - cost[t+1])/n <= 0.01:
                break

        a = np.copy(a1)
        cost = cost/n
        return a, cost

    def projection_optimization(self,KTrain,label_index,m,r,S,trn):
        KK = np.copy(KTrain[label_index,:])
        #http://math.stackexchange.com/questions/158219/is-a-matrix-multiplied-with-its-transpose-something-special
        RM = KK.T.dot(KK)
        Al = np.zeros((m,r))

        for rr in range(r):
            #print rr

            if rr > 0:
                S = S - y.reshape(len(y),1).dot(y.reshape(1,len(y)))

            LM = KK.T.dot(S).dot(KK)
            #U,V = LA.eig(np.matrix(LM),np.matrix(RM))
            U,V = sci.linalg.eig(LM,RM)
            eigenvalue,vector = (list(s) for s in zip(*sorted(zip(U, V), reverse=True)))
            Al[:,rr] = np.copy(vector[0])
            tep = Al[:,rr].T.dot(RM).dot(Al[:,rr])
            Al[:,rr] = np.sqrt(trn/tep)*(Al[:,rr])

            get_vec, cost = self.OptProjectionFast(KK, S, Al[:,rr], 500)
            y = np.float64(KK.dot(Al[:,rr]) > 0)
            y[np.where( y <= 0 )] = -1
            y1 = np.float64(KK.dot(get_vec) > 0)
            y1[np.where( y1 <= 0 )] = -1

            if y1.T.dot(S).dot(y1) >= y.T.dot(S).dot(y):
                Al[:,rr] = np.copy(get_vec)
                y = np.copy(y1)

        Y = np.float32(Al.T.dot(KTrain.T) > 0)
        Y[np.where( Y <= 0 )] = -1

        return Y,Al

    def nearNeighbors(self, elem_bin, elem = None, cant = 0):
        if elem_bin in self.table.keys(): 
            if cant == 0:
                return self.table[elem_bin]
            elif len(self.table[elem_bin]) <= cant:
                return self.table[elem_bin]
            else:
                dist_elem = []
                for elem_buck in self.table[elem_bin]:
                    dist = self.dist.distance(elem_buck['desc'],elem)
                    dist_elem.append((dist,elem_buck))
                dist_elem = sorted(dist_elem, key=itemgetter(0), reverse=True)
                #return dist_elem[:cant]
                return [ elem[1] for elem in dist_elem[:cant] ]
        else:
            #print "No element in buckets"
            return []

    def verificarHash(self,testdata,testgnd, cant = 0):
        buenos = 0
        malos = 0

        KTest = self.sqdist(testdata,self.anchor)
        tY = np.float32(self.Al.T.dot(KTest.T) > 0)
        tY[np.where( tY <= 0 )] = -1

        for i in range(len(testdata)):
            bits = tY[:,i]
            has = ''.join([ '1' if j >= 0 else '0' for j in bits ])

            neighbors = self.nearNeighbors(has, testdata[i], cant )

            good = False
            for ind in neighbors:
                if ind['label'] == testgnd[i]:
                    good = True
                    break

            if good:
                buenos += 1
            else:
                malos += 1
        print "Precision de la tabla Hash: ",float(buenos)/float(buenos + malos)
        return float(buenos)/float(buenos + malos)

    def presRecall(self,has,testdata,testlabel,cant = 0):
        start_time = time.time()

        neighbors = self.nearNeighbors(has,testdata,cant)
        bucket = self.nearNeighbors(has)

        buenas = 0
        malas = 0
        total_buc = 0
        for nei in neighbors:
            if nei['label'] == testlabel:
                buenas += 1
            else:
                malas += 1
        time_total = time.time() - start_time
        for nei in bucket:
            if nei['label'] == testlabel:
                total_buc += 1

        if (buenas + malas) == 0:
            pres = 0
        else:
            pres = float(buenas)/float(buenas+malas)

        if total_buc == 0:
            rec = 0
        else:
            rec = float(buenas)/float(total_buc)

        return pres,rec,time_total

    def lenBucket(self):
        for key in self.table.keys():
            print "Elementos Bucket: ",len(self.table[key])

class SignALSH(object):
    def __init__(self, dimension_array, data, data_train, bits = 3, m = 2, U = 0.75):
        """
        dimension_array: Es la dimension del vector de caracteristicas que se va a utilziar
        bits: es el largo del vector binario
        """
        x = T.dvector('x')
        w = T.dmatrix('W')
        z = T.dot(w,x)
        self.f = function([x, w], z)

        self.m = m
        self.U = U

        self.dist = DistanceMetric(dist = 'sign-alsh')

        self.M = self.getM(data_train)
        self.scale(data)

        self.setW(bits,dimension_array + m)
        self.table = {}

        self.createHash()

    def setW(self, bits, dimension_array):
        self.w = np.random.uniform(-1,1,(bits,dimension_array))

    def getBinaryArray(self, desc):
        x_bin = self.f(desc,self.w)
        return ''.join([ '1' if i >= 0 else '0' for i in x_bin ])

    def createHash(self):
        for elem in self.data:
            x_bin = self.getBinaryArray(elem['desc'])

            if x_bin in self.table.keys():
                self.table[x_bin].append(elem)
            else:
                self.table[x_bin] = [elem]

    def scale(self,data):
        temp = self.U/self.M
        self.data = []
        for vect in data:
            elem = {}
            elem['desc'] = self.P(vect['desc']*temp)
            elem['label'] = vect['label']
            elem['name'] = vect['name']
            self.data.append(elem)

    def getM(self,data):      
        M = LA.norm(data, axis = 1)
        return np.max(M)

    def Q(self,vect):
        return np.concatenate((vect,np.zeros(self.m)),axis=0)

    def P(self,vect):
        temp = []
        norm_x = LA.norm(vect)
        for i in range(1,self.m+1):
            temp.append(1.0/2.0 - norm_x**(2**i))
        return np.concatenate((vect,temp),axis = 0)

    def nearNeighbors(self, elem_bin, elem = None, cant = 0):
        if elem_bin in self.table.keys(): 
            if cant == 0:
                return self.table[elem_bin]
            elif len(self.table[elem_bin]) <= cant:
                return self.table[elem_bin]
            else:
                dist_elem = []
                for elem_buck in self.table[elem_bin]:
                    dist = self.dist.distance(elem_buck['desc'],elem, m=self.m)
                    dist_elem.append((dist,elem_buck))
                dist_elem = sorted(dist_elem, key=itemgetter(0), reverse=True)
                #return dist_elem[:cant]
                return [ elem[1] for elem in dist_elem[:cant] ]
        else:
            #print "No element in buckets"
            return []

    def verificarHash(self,testdata,testgnd, cant = 0):
        buenos = 0
        malos = 0

        for i in range(len(testdata)):
            vect = testdata[i]/LA.norm(testdata[i])
            has = self.getBinaryArray(self.Q(vect))
            neighbors = self.nearNeighbors(has, self.Q(vect), cant)

            good = False
            for ind in neighbors:
                if ind['label'] == testgnd[i]:
                    good = True
                    break

            if good:
                buenos += 1
            else:
                malos += 1
        print "Precision de la tabla Hash: ",float(buenos)/float(buenos + malos)
        #print "Cantidad de buckets usados: ",len(self.table.keys())
        return float(buenos)/float(buenos + malos)

    def presRecall(self,testdata,testlabel,cant = 0):
        start_time = time.time()
        vect = testdata/LA.norm(testdata)
        has = self.getBinaryArray(self.Q(vect))

        neighbors = self.nearNeighbors(has,self.Q(vect),cant)
        bucket = self.nearNeighbors(has)

        buenas = 0
        malas = 0
        total_buc = 0
        for nei in neighbors:
            if nei['label'] == testlabel:
                buenas += 1
            else:
                malas += 1
        time_total = time.time() - start_time
        for nei in bucket:
            if nei['label'] == testlabel:
                total_buc += 1

        if (buenas + malas) == 0:
            pres = 0
        else:
            pres = float(buenas)/float(buenas+malas)

        if total_buc == 0:
            rec = 0
        else:
            rec = float(buenas)/float(total_buc)

        return pres,rec,time_total

    def lenBucket(self):
        for key in self.table.keys():
            print "Elementos Bucket: ",len(self.table[key])

class L2ALSH(object):
    def __init__(self, dimension_array, data, data_train, bits = 3, m = 3, U = 0.83, r = 2.5):
        """
        dimension_array: Es la dimension del vector de caracteristicas que se va a utilziar
        bits: es el largo del vector binario
        """
        x = T.dvector('x')
        w = T.dmatrix('W')
        b = T.dvector('b')
        z = T.dot(w,x) + b
        self.f = function([x, w, b], z)

        self.m = m
        self.U = U
        self.r = r

        self.dist = DistanceMetric(dist = 'eucl')

        self.M = self.getM(data_train)
        self.scale(data)

        self.setW(bits,dimension_array + m)
        self.table = {}

        self.createHash()

    def setW(self, bits, dimension_array):
        self.w = np.random.uniform(-1,1,(bits,dimension_array))
        self.b = np.random.uniform(-1,1,bits)

    def getBinaryArray(self, desc):
        x_bin = np.floor(self.f(desc,self.w,self.b)/self.r)
        return ''.join([ str(i) for i in x_bin  ])

    def createHash(self):
        for elem in self.data:
            x_bin = self.getBinaryArray(elem['desc'])

            if x_bin in self.table.keys():
                self.table[x_bin].append(elem)
            else:
                self.table[x_bin] = [elem]

    def scale(self,data):
        temp = self.U/self.M
        self.data = []
        for vect in data:
            elem = {}
            elem['desc'] = self.P(vect['desc']*temp)
            elem['label'] = vect['label']
            elem['name'] = vect['name']
            self.data.append(elem)

    def getM(self,data):      
        M = LA.norm(data, axis = 1)
        return np.max(M)

    def Q(self,vect):
        return np.concatenate((vect,np.ones(self.m)*0.5),axis=0)

    def P(self,vect):
        temp = []
        norm_x = LA.norm(vect)
        for i in range(1,self.m+1):
            temp.append(norm_x**(2**i))
        return np.concatenate((vect,temp),axis = 0)

    def nearNeighbors(self, elem_bin, elem = None, cant = 0):
        if elem_bin in self.table.keys(): 
            if cant == 0:
                return self.table[elem_bin]
            elif len(self.table[elem_bin]) <= cant:
                return self.table[elem_bin]
            else:
                dist_elem = []
                for elem_buck in self.table[elem_bin]:
                    dist = self.dist.distance(elem_buck['desc'],elem, m=self.m)
                    dist_elem.append((dist,elem_buck))
                dist_elem = sorted(dist_elem, key=itemgetter(0), reverse=True)
                #return dist_elem[:cant]
                return [ elem[1] for elem in dist_elem[:cant] ]
        else:
            #print "No element in buckets"
            return []

    def verificarHash(self,testdata,testgnd, cant = 0):
        buenos = 0
        malos = 0

        for i in range(len(testdata)):
            vect = testdata[i]/LA.norm(testdata[i])
            has = self.getBinaryArray(self.Q(vect))
            neighbors = self.nearNeighbors(has,self.Q(vect),cant)

            good = False
            for ind in neighbors:
                if ind['label'] == testgnd[i]:
                    good = True
                    break

            if good:
                buenos += 1
            else:
                malos += 1
        print "Precision de la tabla Hash: ",float(buenos)/float(buenos + malos)
        #print "Cantidad de buckets usados: ",len(self.table.keys())
        return float(buenos)/float(buenos + malos)

    def presRecall(self,testdata,testlabel,cant = 0):
        start_time = time.time()
        vect = testdata/LA.norm(testdata)
        has = self.getBinaryArray(self.Q(vect))
        neighbors = self.nearNeighbors(has,self.Q(vect),cant)
        bucket = self.nearNeighbors(has)

        buenas = 0
        malas = 0
        total_buc = 0
        for nei in neighbors:
            if nei['label'] == testlabel:
                buenas += 1
            else:
                malas += 1
        time_total = time.time() - start_time
        for nei in bucket:
            if nei['label'] == testlabel:
                total_buc += 1

        if (buenas + malas) == 0:
            pres = 0
        else:
            pres = float(buenas)/float(buenas+malas)

        if total_buc == 0:
            rec = 0
        else:
            rec = float(buenas)/float(total_buc)

        return pres,rec,time_total


    def lenBucket(self):
        for key in self.table.keys():
            print "Elementos Bucket: ",len(self.table[key])
