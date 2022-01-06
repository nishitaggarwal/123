import cv2
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
import PIL.ImageOps
import os,ssl, time

# if (not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl,'_create_unverified_context',None)):
#     ssl._create_default_https_context = ssl._create_unverified_context




X = np.load('image.npz')['arr_0']
y = pd.read_csv('labels.csv')['labels']

classes  = ['A','B','C','D','E','F','G','H','I','J','K','L','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
nclasses = len(classes)


x_train,x_test,y_train,y_test = train_test_split(X,y,random_state = 9,
                                                 train_size = 7500,
                                                 test_size = 2500 )

x_train_scale = x_train/255.0
x_test_scale = x_test/255.0

clf = LogisticRegression(solver = 'saga', multi_class = 'multinomial').fit(x_train_scale,y_train)

y_pred = clf.predict(x_test_scale)
accuracy = accuracy_score(y_test,y_pred)



cap = cv2.VideoCapture(0)

while True:
    ret,frame = cap.read()
    grey = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    height,width = grey.shape
    upper_left = (int(width/2 -56), int(height/2 - 56))
    bottom_right = (int(width/2 + 56), int(height/2 + 56))
    cv2.rectangle(grey,upper_left, bottom_right, (0,255,0), 2 )
    
    roi = grey[upper_left[1]:bottom_right[1] , upper_left[0]:bottom_right[0]]
    im_PIL = Image.fromarray(roi)
    img_bw = im_PIL.convert('L')
    img_bw_resize = img_bw.resize((22,30), Image.ANTIALIAS)
    img_bw_resize_inverted = PIL.ImageOps.invert(img_bw_resize)
    pixel_filter = 20
    min_pixel = np.percentile(img_bw_resize_inverted,pixel_filter)
    img_bw_resize_inverted_scaled = np.clip(img_bw_resize_inverted - min_pixel,0,255)
    max_pixel = np.max(img_bw_resize_inverted)

    img_bw_resize_inverted_scaled = np.asarray(img_bw_resize_inverted_scaled)/max_pixel

    test_sample = np.array(img_bw_resize_inverted_scaled).reshape(1,660)
    test_pred = clf.predict(test_sample)
    print('Predicted Classes', test_pred)

    cv2.imshow('frame',grey)
    if(cv2.waitKey(1) & 0xFF == ord('q') ):
        break

cap.release()
cv2.destroyAllWindows()