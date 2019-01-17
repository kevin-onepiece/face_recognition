import cv2
import numpy as np
import os

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "haarcascade_frontalface_alt.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);
font = cv2.FONT_HERSHEY_SIMPLEX

#iniciate id counter
id = 0
# names related to ids: example ==> Marcelo: id=1,  etc
#names = ['None', 'Marcelo', 'Paula', 'Ilza', 'Z', 'W']

# Initialize and start realtime video capture
cam = cv2.VideoCapture('test/720p.mp4')
cam.set(3, 640) # set video widht
cam.set(4, 680) # set video height

# Define min window size to be recognized as a face
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)
#img=cv2.imread("test/fuqiang0.jpg")


# 判断三帧人名是否相同
def threeissame(img1,img2,img3,img4,img5,img6,img7):  
    standrad_confidence = 30
    
    # 第一帧
    img1 = cv2.flip(img1, -1) # Flip vertically
    gray = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    # 初始的人名集合
    id1_list = []
    # 处理过的人名集合
    id1_list_after = []
    # 初始的人脸坐标集合
    faces_location = []
    # 处理过后的人脸坐标集合
    faces_location_after = []

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        #minSize = (int(minW), int(minH)),
       )

    for(x,y,w,h) in faces:
        id1, confidence = recognizer.predict(gray[y:y+h,x:x+w])

        confidence=confidence-100
        if (confidence < standrad_confidence):
            face_location = (x,y,w,h)
            faces_location.append(face_location)
            id1_list.append(id1)
#        else:
#            return 'unknown'
        
        
    # 第二帧
    img2 = cv2.flip(img2, -1) # Flip vertically
    gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    id2_list = []

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        #minSize = (int(minW), int(minH)),
       )

    for(x,y,w,h) in faces:
        id2, confidence = recognizer.predict(gray[y:y+h,x:x+w])

        confidence=confidence-100
        if (confidence < standrad_confidence):
            id2_list.append(id2)
#        else:
#            return 'unknown'
            
    # 第三帧
    img3 = cv2.flip(img3, -1) # Flip vertically
    gray = cv2.cvtColor(img3,cv2.COLOR_BGR2GRAY)
    id3_list = []

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        #minSize = (int(minW), int(minH)),
       )

    for(x,y,w,h) in faces:
        id3, confidence = recognizer.predict(gray[y:y+h,x:x+w])

        confidence=confidence-100
        if (confidence < standrad_confidence):
            id3_list.append(id3)
#        else:
#            return 'unknown'
            
            
    # 第四帧
    img4 = cv2.flip(img4, -1) # Flip vertically
    gray = cv2.cvtColor(img3,cv2.COLOR_BGR2GRAY)
    id4_list = []

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        #minSize = (int(minW), int(minH)),
       )

    for(x,y,w,h) in faces:
        id4, confidence = recognizer.predict(gray[y:y+h,x:x+w])

        confidence=confidence-100
        if (confidence < standrad_confidence):
            id4_list.append(id4)
            
            
    # 第五帧
    img5 = cv2.flip(img5, -1) # Flip vertically
    gray = cv2.cvtColor(img5,cv2.COLOR_BGR2GRAY)
    id5_list = []

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        #minSize = (int(minW), int(minH)),
       )

    for(x,y,w,h) in faces:
        id5, confidence = recognizer.predict(gray[y:y+h,x:x+w])

        confidence=confidence-100
        if (confidence < standrad_confidence):
            id5_list.append(id5)
            
    
    # 第六帧
    img6 = cv2.flip(img6, -1) # Flip vertically
    gray = cv2.cvtColor(img6,cv2.COLOR_BGR2GRAY)
    id6_list = []

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        #minSize = (int(minW), int(minH)),
       )

    for(x,y,w,h) in faces:
        id6, confidence = recognizer.predict(gray[y:y+h,x:x+w])

        confidence=confidence-100
        if (confidence < standrad_confidence):
            id6_list.append(id6)
            
            
    # 第六帧
    img7 = cv2.flip(img7, -1) # Flip vertically
    gray = cv2.cvtColor(img7,cv2.COLOR_BGR2GRAY)
    id7_list = []

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        #minSize = (int(minW), int(minH)),
       )

    for(x,y,w,h) in faces:
        id7, confidence = recognizer.predict(gray[y:y+h,x:x+w])

        confidence=confidence-100
        if (confidence < standrad_confidence):
            id7_list.append(id7)
        
    
    # 对第一帧的人名集合id1_list，进行判断
    for i in id1_list:
        if i in id2_list:
            if i in id3_list:
                if i in id4_list:
                    if i in id5_list:
                        if i in id6_list:
                            if i in id7_list:  
                                id1_list_after.append(i)
                                faces_location_after.append(faces_location[id1_list.index(i)])
                
                
    return id1_list_after,faces_location_after


while True:
    ret1, img1 = cam.read()
    ret2, img2 = cam.read()
    ret3, img3 = cam.read()
    ret4, img4 = cam.read()
    ret5, img5 = cam.read()
    ret6, img6 = cam.read()
    ret7, img7 = cam.read()
    
    id_list,faces_list = threeissame(img1,img2,img3,img4,img5,img6,img7)
    
    img1 = cv2.flip(img1,-1)
    #for i in id_list:
        
    for (x,y,w,h) in faces_list:
        p=faces_list.index((x,y,w,h))
        cv2.rectangle(img1, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(img1, str(id_list[p]), (x+5,y-5), font, 3, (255,10,255), 2)
#    img = cv2.flip(img, -1) # Flip vertically
#    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#
#    faces = faceCascade.detectMultiScale(
#        gray,
#        scaleFactor = 1.2,
#        minNeighbors = 5,
#        #minSize = (int(minW), int(minH)),
#       )
#
#    for(x,y,w,h) in faces:
#        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
#        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
#
#
#
#        confidence=confidence-100
#        # Check if confidence is less them 100 ==> "0" is perfect match
#        if (confidence < 100):
#            #id = names[id]
#            #confidence = "  {0}%".format(round(confidence))
#            cv2.putText(img, str(id), (x+5,y-5), font, 3, (255,10,255), 2)
##            cv2.putText(img, str(confidence), (x+5,y+h-5), font, 3, (255,255,0), 1)
#        else:
#            id = "unknown"
#            confidence = "  {0}%".format(round(100 - confidence))
#            cv2.putText(img, str(id), (x+5,y-5), font, 3, (255,10,255), 2)



    cv2.imshow('camera',img1)

    k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break

#img=cv2.resize(img,(700,700))
#cv2.imshow("result",img)
cv2.waitKey(0)

#with open('name.txt','w') as f:
  #  f.write(str(id))
# Do a bit of cleanup
#print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()
