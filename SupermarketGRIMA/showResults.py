import cPickle as pickle
import numpy as np
import cv2

def readData():
    data = []
    f=open('./Resultados/panoramica_test.p', 'rb') #
    while 1:
        try:
            data.append(pickle.load(f))
        except EOFError:
            break
    return data

def showImage(data):
    path = '/home/julio/Documents/Dataset/OursDataSupermarket/AMano/'
    width = 200
    hight = 300
    score_min = 0.002
    i = 1
    for elem in data:
        img_label = None
        for label in elem['label_res']:
            img = np.zeros((50,width,3))+255
            cv2.putText(img,str(label), (0,45), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0),2,2)
            if img_label is None:
                img_label = img
            else:
                img_label = np.hstack((img_label,img))

        if elem['score_res'][0] >= score_min:
            img_score = None
            for score in elem['score_res']:
                img = np.zeros((50,width,3))+255
                cv2.putText(img,str(score)[:5], (0,45), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0),2,2)
                if img_score is None:
                    img_score = img
                else:
                    img_score = np.hstack((img_score,img))

            img_img = None
            for name in elem['name_res']:
                #print path + 'crops/' + name.replace('/','')
                name = path + 'random_panoramica/' + name.replace('/','')
                #print name
                img = cv2.resize(cv2.imread(name),(width,hight), interpolation = cv2.INTER_CUBIC)
                if img_img is None:
                    img_img = img
                else:
                    img_img = np.hstack((img_img,img))

            print elem['name'].replace('/','')
            name = path + 'testPanoramica/' + elem['name'].replace('/','')
            img = cv2.resize(cv2.imread(name),(width,hight), interpolation = cv2.INTER_CUBIC)

            img_score_test = np.zeros((50,width,3))+255
            score = elem['name'].replace('/','').split('.')[0].split('_')[1]
            cv2.putText(img_score_test,str(score)[:5], (0,45), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0),2,2)

            h1, w1 = img.shape[:2]
            h5, w5 = img_score_test.shape[:2]
            h2, w2 = img_img.shape[:2]
            h3, w3 = img_score.shape[:2]
            h4, w4 = img_label.shape[:2]
            #h1, w1 = img_label.shape[:2]
            #h2, w2 = img_score.shape[:2]
            #h3, w3 = img_img.shape[:2]
            #h4, w4 = img.shape[:2]
            vis = np.zeros((h1 + h2 + h3 + h4 + h5, max(w1,w2,w3,w4,w5),3), np.uint8)
            #vis[:h1, :w1, :] = img_label
            #vis[h1:h1+h2, :w2, :] = img_score
            #vis[h1+h2:h1+h2+h3, :w3, :] = img_img
            #vis[h1+h2+h3:h1+h2+h3+h4, :w4, :] = img
            vis[:h1, :w1, :] = img
            vis[h1:h1+h5, :w5, :] = img_score_test
            vis[h1+h5:h1+h2+h5, :w2, :] = img_img
            vis[h1+h2+h5:h1+h2+h3+h5, :w3, :] = img_score
            vis[h1+h2+h3+h5:h1+h2+h3+h4+h5, :w4, :] = img_label

            cv2.imwrite(path+'Resultados/panoramica_temp/' + str(i) + '.png',vis)
            #cv2.imshow('Negativas',vis)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()

        i += 1
        
def main():
    data = readData()
    showImage(data)

if __name__ == "__main__":
    main()