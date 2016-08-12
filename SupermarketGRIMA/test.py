import matplotlib.pyplot as plt
import numpy as np
import cPickle as pickle
import time

class Test(object):
    def __init__(self, hashing, exemplar, min_score):
        self.hashing = hashing
        self.exemplar = exemplar
        self.min_score = min_score


    def getTYfromHash(self,data):
        tY_total = []
        for i in range(len(self.hashing)):
            KTest = self.hashing[i].sqdist(data,self.hashing[i].anchor)
            tY = np.float32(self.hashing[i].Al.T.dot(KTest.T) > 0)
            tY[np.where( tY <= 0 )] = -1
            tY_total.append(tY)
        return tY_total

    def getIntersectionNeighbors(self,ksh,neighbors,has):
        #Para cuando se utilicen mas de una tabla
        for elem in ksh.table[has]:
            neighbors.append(elem)
        return neighbors

    def getAccuracy(self,test_data,test_label):
        buenos = 0
        malos = 0
        start_time = time.time()
        tY_total = self.getTYfromHash(test_data)

        for i in range(len(test_data)):
            neighbors = []
            for j in range(len(self.hashing)):
                bits = tY_total[j][:,i]
                has = ''.join([ '1' if t >= 0 else '0' for t in bits ])

                neighbors = self.getIntersectionNeighbors(self.hashing[j],neighbors,has)

            if len(neighbors) == 0:
                print "No hay vecinos"
                malos += 1
            else:
                labels,scores = self.exemplar.getBestScore(test_data[i],neighbors)

                #if scores > 0.08:
                if labels == test_label[i]:
                    buenos += 1
                else:
                    malos += 1

        print("Get accuracy with e-svm--- %s seconds ---" % (time.time() - start_time))

        print "Accuracy: ", float(buenos)/float(buenos + malos)
        print buenos
        print malos
        return float(buenos)/float(buenos + malos)

    def getAccuracyPerClass(self,test_data,test_label,train_all_data):
        tablas_class = {}
        start_time = time.time()
        tY_total = self.getTYfromHash(test_data)

        for i in range(len(test_data)):
            if self.hashing:
                neighbors = []
                for j in range(len(self.hashing)):
                    bits = tY_total[j][:,i]
                    has = ''.join([ '1' if t >= 0 else '0' for t in bits ])

                    neighbors = self.getIntersectionNeighbors(self.hashing[j],neighbors,has)
                label,score = self.exemplar.getBestScore(test_data[i],neighbors)
                if label is None:
                    label,score = self.exemplar.getBestScore(test_data[i],train_all_data)
            else:
                label,score = self.exemplar.getBestScore(test_data[i],train_all_data)

            if score > self.min_score:
                if test_label[i] not in tablas_class.keys():
                    tablas_class[test_label[i]] = np.zeros((2,2))

                if test_label[i] == label:
                    tablas_class[test_label[i]][0,0] += 1
                else:
                    tablas_class[test_label[i]][1,0] += 1
                    if label not in tablas_class.keys():
                        tablas_class[label] = np.zeros((2,2))
                    tablas_class[label][0,1] += 1
        prec_total = 0.0
        recall_total = 0.0
        for i in tablas_class.keys():
            if tablas_class[i][0,1] == 0 and tablas_class[i][0,0] == 0:
                prec = 0
            else:
                prec = tablas_class[i][0,0]/(tablas_class[i][0,0]+tablas_class[i][0,1])
            if tablas_class[i][1,0] == 0 and tablas_class[i][0,0] == 0:
                recall = 0
            else:
                recall = tablas_class[i][0,0]/(tablas_class[i][0,0]+tablas_class[i][1,0])

            #print "Clase: ", self.index_class[i]
            #print tablas_class[i]
            #print "Precision: ", prec
            #print "Recall: ", recall
            prec_total += prec
            recall_total += recall
        print "Final"
        print "Presicion: ", prec_total/len(tablas_class.keys())
        print "Recall: ", recall_total/len(tablas_class.keys())
        print("Get accuracy and recall with e-svm--- %s seconds ---" % (time.time() - start_time))
        return prec_total/len(tablas_class.keys())

    def matrixConfusion(self,test_data,test_label,train_label,train_all_data):
        np.set_printoptions(threshold='nan')
        start_time = time.time()
        tY_total = self.getTYfromHash(test_data)
        matrix = np.zeros((len(np.unique(train_label)),len(np.unique(train_label))))
        for i in range(len(test_data)):
            if self.hashing:
                neighbors = []
                for j in range(len(self.hashing)):
                    bits = tY_total[j][:,i]
                    has = ''.join([ '1' if t >= 0 else '0' for t in bits ])

                    neighbors = self.getIntersectionNeighbors(self.hashing[j],neighbors,has)
                label,score = self.exemplar.getBestScore(test_data[i],neighbors)
                if label is None:
                    #No hay vecinos
                    label,score = self.exemplar.getBestScore(test_data[i],train_all_data)
            else:
                label,score = self.exemplar.getBestScore(test_data[i],train_all_data)
            matrix[test_label[i]-1,label-1] += 1

        plt.matshow(matrix)
        #plt.matshow(matrix[7:36,:])
        plt.colorbar()
        plt.show()
        #getDataFromCM(matrix)
        print("Get Confusion Matrix --- %s seconds ---" % (time.time() - start_time))

    def visualizadorTopClsForTest(self,test_data,test_name,train_all_data,path = '/home/julio/Documents/Dataset/Grocery_products/'):
        start_time = time.time()
        cant = 5
        tma_image = 1000/cant
        tY_total = self.getTYfromHash(test_data)
        for i in range(len(test_data)):
            print path + test_name[i]
            img = cv2.imread(path + test_name[i])
            #img = cv2.resize(img, (500,500), interpolation = cv2.INTER_CUBIC)
            if self.hashing:
                neighbors = []
                for j in range(len(self.hashing)):
                    bits = tY_total[j][:,i]
                    has = ''.join([ '1' if t >= 0 else '0' for t in bits ])

                    neighbors = self.getIntersectionNeighbors(self.hashing[j],neighbors,has)
                maxi_name,mini_name,maxi_score,mini_score = self.exemplar.getTopScore2(test_data[i],neighbors,cant)
            else:
                maxi_name,mini_name,maxi_score,mini_score = self.exemplar.getTopScore2(test_data[i],train_all_data,cant) 
            img_pos = None
            j = 0
            if maxi_score[0] > self.min_score:
                for name in maxi_name:
                    if img_pos is None:
                        name = path +  name
                        print name
                        img_pos = cv2.resize(cv2.imread(name),(tma_image,tma_image), interpolation = cv2.INTER_CUBIC)
                    else:
                        name = path + name
                        print name
                        img2 = cv2.resize(cv2.imread(name),(tma_image,tma_image), interpolation = cv2.INTER_CUBIC)
                        img_pos = np.hstack((img_pos,img2))
                    print "Score: ", maxi_score[j]
                    j += 1
                j = 0
                img_neg = None
                for name in mini_name:
                    if img_neg is None:
                        name = path + name
                        print name
                        img_neg = cv2.resize(cv2.imread(name),(tma_image,tma_image), interpolation = cv2.INTER_CUBIC)
                    else:
                        name = path + name
                        print name
                        img2 = cv2.resize(cv2.imread(name),(tma_image,tma_image), interpolation = cv2.INTER_CUBIC)
                        img_neg = np.hstack((img_neg,img2))
                    print "Score: ", mini_score[j]
                    j += 1
                cv2.imshow('image',img)
                cv2.imshow('Positivas',img_pos)
                cv2.imshow('Negativas',img_neg)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                print "No se cumple el minimo"
        print("Get Top scores (Bad and Good) --- %s seconds ---" % (time.time() - start_time))

    def getBestScoreImage(self,test_data,test_name,name_file):
        tY_total = self.getTYfromHash(test_data)
        f = open( name_file , "wb" )
        start_time = time.time()
        for i,elem in enumerate(test_data):
            res = {}
            neighbors = []
            for j in range(len(self.hashing)):
                bits = tY_total[j][:,i]
                has = ''.join([ '1' if t >= 0 else '0' for t in bits ])

                neighbors = self.getIntersectionNeighbors(self.hashing[j],neighbors,has)

            res['name'] = test_name[i]

            name,score,label = self.exemplar.getTopScore3(elem,neighbors,5)

            res['name_res'] = name
            res['score_res'] = score
            res['label_res'] = label

            pickle.dump( res, f )
        f.close()
        print("Predict all test data--- %s seconds ---" % (time.time() - start_time))
