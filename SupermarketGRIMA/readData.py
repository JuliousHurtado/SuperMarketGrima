import os
import numpy as np
import scipy.io as sio
from sklearn.externals import joblib

class Data(object):
    def __init__(self,db = 'Grozy-120', train_red = True):
        if db == 'Grozy-120':
            path = "/mnt/nas/GrimaRepo/datasets/Grozi-120/"
            self.readIndexData(path)
            self.train_data,self.train_label,self.train_name = self.readDescriptor(path + 'Descriptores/Train/',True)
            self.test_data,self.test_label,self.test_name = self.readDescriptor(path + 'Descriptores/Test_norm/',False)

        if db == 'Chars74K':
            path = "/mnt/nas/GrimaRepo/datasets/Chars74K/"
            self.readDescriptorChars(path + 'Descriptor/')

        if db == 'GroceryProducts':
            self.list_label_test = [9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35]
            self.train_red = train_red
            path = "/mnt/nas/GrimaRepo/datasets/Grocery_products/"
            list_classes_index = "Training/TrainingClassesIndex.mat"
            self.readIndexClases(path + list_classes_index)
            #self.train_data,self.train_label,self.train_name =  self.readDescMatFC6GP(path + 'Descriptores_new/Train/',True)
            #self.test_data,self.test_label,self.test_name =  self.readDescMatFC6GP(path + 'Descriptores_new/Test/',False)
            self.train_data,self.train_label,self.train_name =  self.readDescMatFC6GP(path + 'new_testing/descriptor_train/',True)
            self.test_data,self.test_label,self.test_name =  self.readDescMatFC6GP(path + 'new_testing/descriptor/',False)

        if db == 'SupermarketGRIMA':
            path = '/mnt/nas/GrimaRepo/datasets/SupermarketGRIMA/ImageRodrigo/FC6/'
            sep = True
            if sep:
                self.train_data,self.train_label,self.train_name,self.test_data,self.test_label,self.test_name = self.readSupermarketFC6(path,sep)
            else:
                self.train_data,self.train_label,self.train_name = self.readSupermarketFC6(path,sep)



        self.path_save_models = '/mnt/nas/GrimaRepo/jahurtado/'

    def readIndexClases(self,path):
        self.index_class = {}
        data = sio.loadmat(path)
        for i in range(len(data['classes'][0])):
            self.index_class[data['classes'][0][i][0]] = data['indices'][0][i]

    def readDescMatFC6GP(self,path,Train):
        data = []
        label = []
        nombre = []

        s = '/'
        i = 0
        for elem in os.listdir(path):
            i += 1
            dict = {}
            desc = sio.loadmat(path + elem)
            if 'stored' in desc.keys() and (np.isnan(desc['stored']).sum() == 0) and (np.isinf(desc['stored']).sum() == 0):
                if Train:
                    name = s.join(desc['file'][0].split('/')[8:-1])
                    #print name
                    #print desc['file'][0]
                    if self.train_red:
                        if self.index_class[name] in self.list_label_test:
                            label.append(self.index_class[name])
                            nombre.append(s.join(desc['file'][0].split('/')[5:]))
                            data.append(desc['stored'][0])
                    else:
                        label.append(self.index_class[name])
                        nombre.append(s.join(desc['file'][0].split('/')[5:]))
                        data.append(desc['stored'][0])
                else:
                    if desc['file'][0].split('/')[-2] != 'images':
                        label.append(int(desc['file'][0].split('/')[-2]))
                        nombre.append(s.join(desc['file'][0].split('/')[5:]))
                        data.append(desc['stored'][0])
        return np.array(data),np.array(label),nombre

    def readIndexData(self,path):
        self.index_class = {}
        path = path + 'index2/UPC_index.txt'
        f = open(path,'r')
        lines = f.readlines()
        for i in range(len(lines)):
            if len(lines[i].strip()) > 0:
                elem = lines[i].strip()
                if len(elem) < 4:
                    self.index_class[int(elem)] = lines[i+2].strip()

    def readDescriptor(self,path,train):
        s = '/'
        data = []
        label = []
        name = []
        for elem in os.listdir(path):
            desc = sio.loadmat(path + elem)
            if np.isnan(desc['stored']).sum() == 0 and np.isinf(desc['stored']).sum() == 0:
                if train:
                    label.append(int(desc['file'][0].split('/')[-4]))
                    name.append(s.join(desc['file'][0].split('/')[-4:]))
                else:
                    label.append(int(desc['file'][0].split('/')[-3]))
                    name.append(s.join(desc['file'][0].split('/')[-4:]))
                data.append(desc['stored'][0])
        return data,label,name

    def readDescriptorChars(self, path):
        self.train_data = []
        self.train_label = []
        self.train_name = []

        self.test_data = []
        self.test_label = []
        self.test_name = []

        s = '/'
        for elem in os.listdir(path):
            desc = sio.loadmat(path + elem)
            if np.isnan(desc['stored']).sum() == 0 and np.isinf(desc['stored']).sum() == 0:
                label = int(desc['file'][0].split('/')[-2][-3:])
                name = s.join(desc['file'][-4:])
                data = desc['stored'][0]
                if self.test_label.count(label) < 20:
                    self.test_data.append(data)
                    self.test_label.append(label)
                    self.test_name.append(name)
                else:
                    self.train_data.append(data)
                    self.train_label.append(label)
                    self.train_name.append(name)

    def subconjuntoDatos(self,train,cant):
        if train:
            data_all = self.train_data
            label_all = self.train_label
            name_all = self.train_name
        else:
            data_all = self.test_data
            label_all = self.test_label
            name_all = self.test_name
            
        data = np.array([])
        label = np.array([])
        name = []
        for i in range(len(data_all)):
            if len(label[np.where( label == label_all[i] )]) < cant:
                if len(name) == 0:
                    data = np.array([data_all[i]])
                    label = np.array([label_all[i]])
                    name = [name_all[i]]
                else:
                    data = np.append(data,[data_all[i]],axis = 0)
                    label = np.append(label,label_all[i])
                    name.append(name_all[i])
        return data,label,name

    def getTrain(self):
        return self.train_data, self.train_label, self.train_name

    def getTest(self):
        return self.test_data, self.test_label, self.test_name 

    def getExtra(self):
        return self.data_extra,self.label_extra

    def readExtraProducts(self):
        path = "/mnt/nas/GrimaRepo/datasets/Grozi-120/"
        self.data_extra,self.label_extra =  self.readDescMatFC62(path + 'Descriptores/Train/')
        self.data_extra,self.label_extra =  self.readDescMatFC62(path + 'Descriptores/Test_norm/', self.data_extra, self.label_extra)

    def readDescMatFC62(self,path,data = [],label = []):
        s = '/'
        for elem in os.listdir(path):
            desc = sio.loadmat(path + elem)
            #print desc
            if np.isnan(desc['stored']).sum() == 0 and np.isinf(desc['stored']).sum() == 1:
                data.append(desc['stored'][0])
                label.append(1)
            if len(data) > 500:
                break
        return data,label

    def saveModels(self,clf,file):
        s = '/'
        if not os.path.exists(self.path_save_models + s.join(file.split('/')[:-1])):
            os.makedirs(self.path_save_models + s.join(file.split('/')[:-1]))
        #print self.path_save_models + file
        joblib.dump(clf, self.path_save_models + file)

    def loadModels(self,file):
        return joblib.load(file)

    def oneDescriptor(self):
        arch = ['0.fc6.mat','1.fc6.mat']

        data = []
        label = []
        nombre = []
        for elem in arch:
            desc = sio.loadmat(elem)
            label.append(0)
            nombre.append(desc['file'][0])
            data.append(desc['stored'][0])
        
        return data,label,nombre

    def saveMat(self,data,label,name,Train):
        if Train:
            path = '/mnt/nas/GrimaRepo/datasets/Grocery_products/new_testing/descriptorPCA/Train/'
        else:
            path = '/mnt/nas/GrimaRepo/datasets/Grocery_products/new_testing/descriptorPCA/Test/'

        for i,elem in enumerate(data):
            a_dict = {'desc':elem, 'label':label[i], 'name':name[i]}
            sio.savemat(path + str(i) + '.mat', {'data': a_dict})

    def loadMat(self,Train):
        if Train:
            path = '/mnt/nas/GrimaRepo/datasets/Grocery_products/new_testing/descriptorPCA/Train/'
        else:
            path = '/mnt/nas/GrimaRepo/datasets/Grocery_products/new_testing/descriptorPCA/Test/'

        data = []
        label = []
        nombre = []

        for elem in os.listdir(path):
            desc = sio.loadmat(path + elem)
            data.append(desc['data']['desc'][0][0][0])
            label.append(desc['data']['label'][0][0][0][0])
            nombre.append(desc['data']['name'][0][0][0])

        return np.array(data),np.array(label),nombre

    def readSupermarketFC6(self,path,sep = False):
        data_temp = []
        label_temp = []
        name_temp = []
        s = '/'
        for elem in os.listdir(path):
            desc = sio.loadmat(path + elem)
            if np.isnan(desc['stored']).sum() == 0 and np.isinf(desc['stored']).sum() == 0:
                label_temp.append(int(desc['file'][0].split('/')[-1].split('_')[0]))
                name_temp.append(s.join(desc['file'][0].split('/')[-1]))
                data_temp.append(desc['stored'][0])

        data = []
        label = []
        name = []
        for i,elem in enumerate(data_temp):
            if label_temp.count(label_temp[i]) > 1:
                data.append(elem)
                name.append(name_temp[i])
                label.append(label_temp[i])

        if sep:
            data_train = []
            label_train = []
            name_train = []
            data_test = []
            label_test = []
            name_test = []
            for i,elem in enumerate(data):
                if label_test.count(label[i]) < float(label.count(label[i]))*0.2:
                    data_test.append(elem)
                    name_test.append(name[i])
                    label_test.append(label[i])
                else:
                    data_train.append(elem)
                    name_train.append(name[i])
                    label_train.append(label[i])
            return np.array(data_train),np.array(label_train),name_train,np.array(data_test),np.array(label_test),name_test
        else:
            return np.array(data),np.array(label),name
           
