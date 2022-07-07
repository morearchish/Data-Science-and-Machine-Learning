import cv2
import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier

cap =cv2.VideoCapture(0)
detector = cv2.CascadeClassifier("D:\data\haarcascade_frontalface_default.xml.txt")
name=input("Enter your name : ")
frames=[]
outputs=[]

data= np.load("face_data.npy")
print(data.shape,data.dtype)
X=data[:,1:].astype(int)
y=data[:,0]

model=KNeighborsClassifier()
model.fit(X,y)





while True:
    ret,frame=cap.read()

    if ret:

        faces=detector.detectMultiScale(frame)

        for face in faces:

            x,y,w,h = face
            cut=frame[y:y+h , x:x+w]
            fix=cv2.resize(cut,(100,100))
            gray=cv2.cvtColor(fix,cv2.COLOR_BGR2GRAY)
            out=model.predict([gray.flatten()])
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            cv2.putText(frame,str(out[0]),(x,y-10),cv2.FONT_HERSHEY_COMPLEX,(255,0,0),2)
            print(out)
            cv2.imshow("My Face",gray)


        cv2.imshow("My Screen",frame)


    key=cv2.waitKey(1)

    if key==ord("q"):
        break

    if key==ord("c"):
        cv2.imwrite(name+".jpg",frame)
        frames.append(gray.flatten())
        outputs.append([name])


X=np.array(frames)
y=np.array(outputs)
data=np.hstack([y,X])


f_name="face_data.npy"
if os.path.exists(f_name):
    old=np.load(f_name)
    data=np.vstack([old,data])
np.save(f_name,data)



cap.release()
cv2.destroyAllWindows()
