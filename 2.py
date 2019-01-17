import numpy as np
from PIL import Image
import os
import cv2
 
# Path for face image database
path = 'dataset'
 
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
 
# function to get the images and label data
def getImagesAndLabels(path):
    c = 1
    X = []   #人脸图片特征向量
    y = []   #每张图片标签,和人名一一对应
    name = dict()
    for dirname, dirnames, filenames in os.walk('dataset'):
        for subdirname in dirnames:
            subject_path = os.path.join(dirname, subdirname)
            for filename in os.listdir(subject_path):
                #读取图片并调整大小
                im = cv2.imread(os.path.join(subject_path, filename), cv2.IMREAD_GRAYSCALE)
                im = cv2.resize(im, (100,100))
#                faces = detector.detectMultiScale(im)
                X.append(np.asarray(im, dtype=np.uint8))
                y.append(c)
               
            name[c] = subdirname
            
            c += 1

    return X,y
 
print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")
faces,ids = getImagesAndLabels(path)
print(np.array(ids))
recognizer.train(faces, np.array(ids))
 
# Save the model into trainer/trainer.yml
recognizer.write('trainer/trainer_test.yml') # recognizer.save() worked on Mac, but not on Pi
 
# Print the numer of faces trained and end program
print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))