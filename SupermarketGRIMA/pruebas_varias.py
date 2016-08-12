from sklearn import svm
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.calibration import CalibratedClassifierCV
from sklearn.externals import joblib
from sklearn.decomposition import PCA
from sklearn.preprocessing import Normalizer
import numpy as np
import scipy.io as sio
import os
import time, random
import matplotlib.pyplot as plt
import cPickle as pickle

from ksh import KSH
from exemplar import Exemplar
from lsh import LSH
from readData import Data

def procesamientoDatos(train,test,extra = None):
    norm = Normalizer()
    train = norm.fit(train).transform(train)
    test = norm.transform(test)

    pca = PCA(n_components=3000)
    X_r = pca.fit(train).transform(train)
    Y_r = pca.transform(test)
    if extra is not None:
        return X_r,Y_r,pca.transform(extra)
        
    return X_r,Y_r

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    #tick_marks = np.arange(len(iris.target_names))
    #plt.xticks(tick_marks, iris.target_names, rotation=45)
    #plt.yticks(tick_marks, iris.target_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def metrics(testgnd,predicted):
    print classification_report(testgnd, predicted)
    cm = confusion_matrix(testgnd, predicted)
    np.set_printoptions(precision=2)
    print('Confusion matrix, without normalization')
    print(cm)
    plt.figure()
    plot_confusion_matrix(cm)

def oneClassifier(traindata,traingnd,testdata,testgnd):

    start_time = time.time()
    #linear_svm = svm.SVC(C = 2048, kernel='rbf', gamma=2, decision_function_shape='ovr')
    linear_svm = svm.LinearSVC(C=10, verbose=False)
    linear_svm.fit(traindata,traingnd)
    print("Create and Train Classifier --- %s seconds ---" % (time.time() - start_time))

    start_time = time.time()
    print(linear_svm.score(testdata,testgnd))
    predicted = linear_svm.predict(testdata)
    metrics(testgnd,predicted)
    print("Predict all test data--- %s seconds ---" % (time.time() - start_time))

def onePerClassClassifier(traindata,traingnd,testdata,testgnd):

    start_time = time.time()
    cls = {}
    for i in np.unique(traingnd):
        train = []
        label = []
        for j in range(len(traindata)):
            if traingnd[j] == i:
                label.append(0)
            else:
                label.append(1)
        class_weight = {0 : 0.5, 1 : 0.01}
        linear_svm = svm.LinearSVC(C=10, verbose=False, class_weight = class_weight)
        #linear_svm = svm.SVC(C = 2048, kernel='rbf', gamma=2)
        #linear_svm.fit(traindata,label)
        cal_cls = CalibratedClassifierCV(linear_svm, cv=2, method='sigmoid')
        cal_cls.fit(traindata,label)
        cls[i] = cal_cls
    print("Create and Train Classifier --- %s seconds ---" % (time.time() - start_time))

    start_time = time.time()
    buenos = 0
    malos = 0
    for i in range(len(testdata)):
        max_proba = 0
        label_predict = None
        for elem in cls.keys():
            predict = cls[elem].predict_proba(testdata[i].reshape(1, -1))
            if predict[0][0] > max_proba:
                max_proba = predict[0][0]
                label_predict = elem
        if label_predict == testgnd[i]:
            buenos += 1
        else:
            malos += 1

    print "Accuracy: ", float(buenos)/float(buenos + malos)
    print("Predict all test data--- %s seconds ---" % (time.time() - start_time))

def onePerBucket(traindata,traingnd,testdata,testgnd,trainName):
    data = []
    for i in range(len(traindata)):
        data.append({'desc': traindata[i] , 'label': traingnd[i]})

    start_time = time.time()
    cont = True
    while cont:
        ksh = KSH(traindata,traingnd,data, m = 100, r = 2, trn = 600)
        if ksh.verificarHash(testdata,testgnd) > 0.7:
            cont = False

    cls = {}
    for key in ksh.table.keys():
        train = []
        label = []
        for elem in ksh.table[key]:
            train.append(elem['desc'])
            label.append(elem['label'])

        linear_svm = svm.LinearSVC(C=1, verbose=False)
        linear_svm.fit(train,label)

        cls[key] = linear_svm
    print("Create and Train Classifier and Table--- %s seconds ---" % (time.time() - start_time))


    start_time = time.time()
    buenos = 0
    malos = 0
    KTest = ksh.sqdist(testdata,ksh.anchor)
    tY = np.float32(ksh.Al.T.dot(KTest.T) > 0)
    tY[np.where( tY <= 0 )] = -1
    for i in range(len(testdata)):
        bits = tY[:,i]
        has = ''.join([ '1' if t >= 0 else '0' for t in bits ])

        if has in cls.keys():
            label_predict = cls[has].predict(testdata[i].reshape(1,-1))
            if label_predict == testgnd[i]:
                buenos += 1
            else:
                malos += 1
        else:
            malos += 1

    print "Accuracy: ", float(buenos)/float(buenos + malos)
    print("Predict all test data--- %s seconds ---" % (time.time() - start_time))

def onePerExemplarPerBucket(traindata,traingnd,testdata,testgnd,trainName):
    data = []
    exemplar = Exemplar(porct = 1,HM = True,cal = True)
    for i in range(len(traindata)):
        data.append({'desc': traindata[i] , 'label': traingnd[i]})

    start_time = time.time()
    cont = True
    while cont:
        ksh = KSH(traindata,traingnd,data, m = 100, r = 2, trn = 600)
        #ksh = LSH(len(data[0]['desc']),data)
        if ksh.verificarHash(testdata,testgnd) > 0.7:
            cont = False
        ksh.lenBucket()

    for key in ksh.table.keys():
        train = []
        label = []
        for elem in ksh.table[key]:
            train.append(elem['desc'])
            label.append(elem['label'])

        for elem in ksh.table[key]:
            elem['cls'] = exemplar.getCls(elem,np.array(train),np.array(label))
    print("Create and Train Classifier and Table--- %s seconds ---" % (time.time() - start_time))

    start_time = time.time()
    buenos = 0
    malos = 0
    KTest = ksh.sqdist(testdata,ksh.anchor)
    tY = np.float32(ksh.Al.T.dot(KTest.T) > 0)
    tY[np.where( tY <= 0 )] = -1
    for i in range(len(testdata)):
        bits = tY[:,i]
        has = ''.join([ '1' if t >= 0 else '0' for t in bits ])
        #has = ksh.getBinaryArray(testdata[i])
        neighbors = ksh.nearNeighbors(has)

        if len(neighbors) > 0:
            label_predict,score = exemplar.getBestScore(testdata[i],neighbors)
            if label_predict == testgnd[i]:
                buenos += 1
            else:
                malos += 1
        else:
            malos += 1

    print "Accuracy: ", float(buenos)/float(buenos + malos)
    print("Predict all test data--- %s seconds ---" % (time.time() - start_time))

def vecinoCercano(train_data,train_label,train_name,test_data,test_label,test_name):
    data = []
    for i,elem in enumerate(train_data):
        data.append({'desc': elem , 'label': train_label[i], 'file_name':train_name[i][-1], 'real_name': train_name[i]})

    print("Partieron")
    start_time = time.time()
    cont = True
    while cont:
        ksh = KSH(train_data,train_label,data, m = 2000, r = 4, trn = 2000)
        if ksh.verificarHash(test_data,test_label) > 0.9:
            cont = False
    ksh.lenBucket()
    print("Bucket created--- %s seconds ---" % (time.time() - start_time))

def lessExample():
    arch = ['0.fc6.mat','1.fc6.mat']
    la = [32,31] #rice 32, pasta 31
    name = ['13.jpg','62.jpg']

    clfs = []
    data_extra = []
    for i in range(len(arch)):
        desc = sio.loadmat(arch[i])
        data_extra.append(desc['stored'][0])
        clfs.append({'label': la[i] , 'name':desc['file'][0], 'file_name':name[i]})

    read_data = Data(db = 'GroceryProducts')
    train_data,train_label,train_name = read_data.subconjuntoDatos(True,10)
    test_data,test_label,test_name = read_data.readDescMatFC6GP("/mnt/nas/GrimaRepo/datasets/Grocery_products/new_testing/descriptor/",False)
    
    
    print "Data read"
    train_data,test_data,data_extra = procesamientoDatos(train_data,test_data,np.array(data_extra))
    print "Procesada"

    for i,elem in enumerate(clfs):
        elem['desc'] = data_extra[i]
        
    data = []
    for i,elem in enumerate(train_data):
        data.append({'desc': elem , 'label': train_label[i], 'file_name':train_name[i][-1]})

    cont = True
    while cont:
        ksh = KSH(train_data,train_label,data, m = 100, r = 2, trn = 200)
        if ksh.verificarHash(test_data,test_label) > 0.7:
            cont = False
    ksh.lenBucket()

    KTest = ksh.sqdist(np.array(data_extra),ksh.anchor)
    tY = np.float32(ksh.Al.T.dot(KTest.T) > 0)
    tY[np.where( tY <= 0 )] = -1
    exemplar = Exemplar( HM = True, cal = True, peso_0 = 1, peso_1 = 0.01)
    for i,elem in enumerate(clfs):
        bits = tY[:,i]
        has = ''.join([ '1' if t >= 0 else '0' for t in bits ])
        elem['cls'] = exemplar.getCls(elem,train_data,train_label,ksh.table[has])
        elem['pred'] = []
        elem['score'] = []
    print "Classifier created"


    for i,elem in enumerate(test_data):
        for clf in clfs:
            scr = clf['cls'].predict_proba(elem.reshape(1,-1))
            if len(clf['score']) < 5:
                clf['pred'].append(test_name[i])
                clf['score'].append(scr[0][0])
                clf['score'],clf['pred'] = (list(t) for t in zip(*sorted(zip(clf['score'],clf['pred']), reverse=True)))
            elif scr[0][0] >= clf['score'][-1]:
                clf['pred'][-1] = test_name[i]
                clf['score'][-1] = scr[0][0]
                clf['score'],clf['pred'] = (list(t) for t in zip(*sorted(zip(clf['score'],clf['pred']), reverse=True)))

    print clfs[0]['pred']
    print clfs[0]['score']
    print clfs[1]['pred']
    print clfs[1]['score']

def onePerExemplar(train_data,train_label,train_name,test_data,test_label,test_name):
    print('No Hash')

    data = []
    exemplar = Exemplar( HM = True, cal = True, peso_0 = 1, peso_1 = 0.01)
    for i,elem in enumerate(train_data):
        data.append({'desc': elem , 'label': train_label[i], 'file_name':train_name[i][-1], 'real_name': train_name[i]})

    start_time = time.time()
    for i,elem in enumerate(data):
        elem['cls'] = exemplar.getCls(elem,train_data,train_label,data)
    print("Classifier created --- %s seconds ---" % (time.time() - start_time))

    f = open( "resultados_no_Hash.p", "wb" )
    for i,elem in enumerate(test_data):
        res = {}

        res['name'] = test_name[i]

        name,score,label = exemplar.getTopScore3(elem,data,5)

        res['name_res'] = name
        res['score_res'] = score
        res['label_res'] = label

        pickle.dump( res, f )
    f.close()
    print("Predict all test data--- %s seconds ---" % (time.time() - start_time))

def funcional(train_data,train_label,train_name,test_data,test_label,test_name):
    print('Hash')
        
    data = []
    for i,elem in enumerate(train_data):
        data.append({'desc': elem , 'label': train_label[i], 'file_name':train_name[i][-1], 'real_name': train_name[i]})

    start_time = time.time()
    cont = True
    while cont:
        ksh = KSH(train_data,train_label,data, m = 500, r = 2, trn = 2000)
        if ksh.verificarHash(test_data,test_label) > 0.9:
            cont = False
    ksh.lenBucket()
    print("Bucket created--- %s seconds ---" % (time.time() - start_time))

    start_time = time.time()
    KTest = ksh.sqdist(np.array(train_data),ksh.anchor)
    tY = np.float32(ksh.Al.T.dot(KTest.T) > 0)
    tY[np.where( tY <= 0 )] = -1
    exemplar = Exemplar( HM = True, cal = True, peso_0 = 1, peso_1 = 0.01)
    for i,elem in enumerate(data):
        bits = tY[:,i]
        has = ''.join([ '1' if t >= 0 else '0' for t in bits ])
        elem['cls'] = exemplar.getCls(elem,train_data,train_label,data)
    print("Classifier created--- %s seconds ---" % (time.time() - start_time))


    KTest = ksh.sqdist(np.array(test_data),ksh.anchor)
    tY = np.float32(ksh.Al.T.dot(KTest.T) > 0)
    tY[np.where( tY <= 0 )] = -1
    f = open( "resultados_no_Hash_in_Train_r_4_new.p", "wb" )
    start_time = time.time()
    for i,elem in enumerate(test_data):
        res = {}
        bits = tY[:,i]
        has = ''.join([ '1' if t >= 0 else '0' for t in bits ])

        res['name'] = test_name[i]

        name,score,label = exemplar.getTopScore3(elem,ksh.table[has],5)

        res['name_res'] = name
        res['score_res'] = score
        res['label_res'] = label

        pickle.dump( res, f )
    f.close()
    print("Predict all test data--- %s seconds ---" % (time.time() - start_time))

def lshFuncional(train_data,train_label,train_name,test_data,test_label,test_name):
    data = []
    for i,elem in enumerate(train_data):
        data.append({'desc': elem , 'label': train_label[i], 'file_name':train_name[i][-1], 'real_name': train_name[i]})

    start_time = time.time()
    cont = True
    while cont:
        lsh = LSH(3000,data,2)
        if lsh.verificarHash(test_data,test_label) > 0.9:
            cont = False
    lsh.lenBucket()
    print("Bucket created--- %s seconds ---" % (time.time() - start_time))

    start_time = time.time()
    exemplar = Exemplar( HM = True, cal = True, peso_0 = 1, peso_1 = 0.01)
    for i,elem in enumerate(data):
        elem['cls'] = exemplar.getCls(elem,train_data,train_label,data)
    print("Classifier created--- %s seconds ---" % (time.time() - start_time))

    f = open( "resultados_no_Hash_in_Train_lsh_2.p", "wb" )
    start_time = time.time()
    for i,elem in enumerate(test_data):
        res = {}
        has = lsh.getBinaryArray(elem)

        res['name'] = test_name[i]

        name,score,label = exemplar.getTopScore3(elem,lsh.table[has],5)

        res['name_res'] = name
        res['score_res'] = score
        res['label_res'] = label

        pickle.dump( res, f )
    f.close()
    print("Predict all test data--- %s seconds ---" % (time.time() - start_time))

def cluster(train_data,train_label,train_name,test_data,test_label,test_name):
    from sklearn.cluster import MiniBatchKMeans
    import scipy.spatial.distance

    print('Learning the dictionary... ')
    rng = np.random.RandomState(0)
    kmeans = MiniBatchKMeans(n_clusters=40, random_state=rng, verbose=True)
    
    buffer = []
    index = 1
    t0 = time.time()

    # The online learning part: cycle over the whole dataset 6 times
    index = 0
    for _ in range(6):
        for elem in train_data:
            buffer.append(elem.reshape(1,-1))
            index += 1
            if index % 100 == 0:
                data = np.concatenate(buffer, axis=0)
                #data -= np.mean(data, axis=0)
                #data /= np.std(data, axis=0)
                kmeans.partial_fit(data)
                buffer = []
            if index % 1000 == 0:
                print('Partial fit of %4i out of %i'
                      % (index, 6 * len(train_data)))

    dt = time.time() - t0
    print('done in %.2fs.' % dt)

    print kmeans.cluster_centers_.shape

    for cen in kmeans.cluster_centers_:
        mini = None
        maxi = None
        name = None
        for i,elem in enumerate(train_data):
            #elem -= np.mean(elem,axis=0)
            #elem /= np.std(elem,axis=0)
            dist = np.linalg.norm(elem.reshape(1,-1)-cen)
            if  dist < mini or mini is None:
                name = train_name[i]
                mini = dist
            if dist > maxi or maxi is None:
                maxi = dist
        print name
        print mini
        print maxi

def main():
    """
    Primero hacer la tabla hash, luego hacer los clasificadores de cada elemento de train
    """
    #traindata,traingnd,testdata,testgnd,trainName = readData()
    #cant_subconjunto = 20
    #traindata,traingnd,testdata,testgnd,trainName = subconjuntoDatos(traindata,traingnd,testdata,testgnd,trainName,cant_subconjunto)
    #traindata,testdata = procesamientoDatos(traindata,testdata)
    
    #onePerClassClassifier(traindata,traingnd,testdata,testgnd)
    #onePerBucket(traindata,traingnd,testdata,testgnd,trainName)
    #onePerExemplarPerBucket(traindata,traingnd,testdata,testgnd,trainName)
    #vecinoCercano(traindata,traingnd,testdata,testgnd,trainName)
    #lessExample()

    read_data = Data(db = 'algo')
    #train_data,train_label,train_name = read_data.getTrain()
    #test_data,test_label,test_name = read_data.getTest()
    #train_data,train_label,train_name = read_data.subconjuntoDatos(True,10)
    #test_data,test_label,test_name = read_data.subconjuntoDatos(False,10)
    train_data,train_label,train_name = read_data.loadMat(True)
    test_data,test_label,test_name = read_data.loadMat(False)
    print "Data read"

    #train_data,test_data = procesamientoDatos(train_data,test_data)
    #print "Procesada"

    #read_data.saveMat(train_data,train_label,train_name,True)
    #read_data.saveMat(test_data,test_label,test_name,False)
    
    print(train_data.shape)
    print(test_data.shape)

    #oneClassifier(train_data,train_label,test_data,test_label)
    #onePerExemplar(train_data,train_label,train_name,test_data,test_label,test_name)
    #funcional(train_data,train_label,train_name,test_data,test_label,test_name)
    #lshFuncional(train_data,train_label,train_name,test_data,test_label,test_name)
    #vecinoCercano(train_data,train_label,train_name,test_data,test_label,test_name)
    cluster(train_data,train_label,train_name,test_data,test_label,test_name)

if __name__ == "__main__":
    main()

"""
The C parameter tells the SVM optimization how much you want to avoid misclassifying 
each training example. For large values of C, the optimization will choose a smaller-margin 
hyperplane if that hyperplane does a better job of getting all the training points 
classified correctly. Conversely, a very small value of C will cause the optimizer to look 
for a larger-margin separating hyperplane, even if that hyperplane misclassifies more points. 
For very tiny values of C, you should get misclassified examples, often even if your training
 data is linearly separable.
"""