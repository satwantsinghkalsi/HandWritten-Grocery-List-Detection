# Import the modules
import cv2
from sklearn.externals import joblib
from skimage.feature import hog
import numpy as np
import argparse
from sklearn.svm import LinearSVC
from collections import Counter
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier as knn
import os
import fnmatch
from sys import argv
def accuracy(pred_label,true_label):
    count=0.0
    if len(pred_label)!=len(true_label):
	return 0
    else:
        for i in range(0,len(pred_label)):
            if pred_label[i].lower()==true_label[i].lower():
               count+=1
    return count/len(true_label)
# Load the classifier
clf=None
if argv[1]=="KNN":
   clf = joblib.load("chars_cls_"+argv[1]+".pkl")
else:
   clf = joblib.load("chars_cls_SVM.pkl")
y_true=None
y_pred=[]
list_hog_fd = []
responses = []
# Read the input image
for dirpath, dirs, files in os.walk('/mnt/c/Users/satwa/Documents/CS6375ML/FinalProject/Handwritten-Grocery-List-Detection/tst'):
    for filename in fnmatch.filter(files, '*.jpg'):
        im = cv2.imread(dirpath+"/"+filename)
        
        # Convert to grayscale and apply Gaussian filtering
        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)
        #out = np.zeros(im.shape,np.uint8)
        # Threshold the image
        im_th = cv2.adaptiveThreshold(im_gray,255,1,1,11,2)
        #ret, im_th = cv2.threshold(im_gray, 90, 255, cv2.THRESH_BINARY_INV)
        
        def sort_contours(cnts, method="left-to-right"):
	    # initialize the reverse flag and sort index
            reverse = False
            i = 0 
	    # construct the list of bounding boxes and sort them from left to right
            boundingBoxes = [cv2.boundingRect(c) for c in cnts]
            (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes), key=lambda b:b[1][i], reverse=reverse))
 
	    # return the list of sorted contours and bounding boxes
            return (cnts, boundingBoxes)

        # Find contours in the image
        _, ctrs, hier = cv2.findContours(im_th,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        #sort 
        ctrs,rects =  sort_contours(ctrs)
        
        # Get rectangles contains each contour
        rects = [cv2.boundingRect(ctr) for ctr in ctrs]
        
        # For each rectangular region, calculate HOG features and predict
        # the digit using Linear SVM.
        result = []
        for cnt in ctrs:
            #print (cv2.contourArea(cnt))
            if cv2.contourArea(cnt)>0:
                [x,y,w,h] = cv2.boundingRect(cnt)
                if  h>0:
                    cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),3)
                    roi = im_th[y:y+h,x:x+w]
                    roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
                    roi = cv2.dilate(roi, (3, 3))
                    roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
                    nbr = clf.predict(np.array([roi_hog_fd],"float32"))
                    #print(chr(nbr))
                    #cv2.imshow('im',im)
                    #cv2.waitKey(0)
                    #print (chr(nbr)) 
                    result.append(chr(nbr))
        end=""
        for nbrs in result:
	    y_pred.append(nbrs)
            print nbrs, 
        #cv2.imshow('im',im)
        cv2.waitKey(0)
	print
file = open("/mnt/c/Users/satwa/Documents/CS6375ML/FinalProject/Handwritten-Grocery-List-Detection/inputimage/"+argv[2],"r")
for line in file:
    y_true=line.split()
#print y_true
#print y_pred	
#y_true=['A', 'p','p','l','e','2','K','i','l','o','m','a','n','g','o','3','l','b','c','u','c','u','m','b','e','r','5']
#y_pred=['A','P','P','L','e','2','K','1','L','0','m','a','n','G','o','S','k','b','C','U','C','U','m','b','e','R','5']
#y_true=['O','n','i','O','n','3','L','b','S','P','I','n','a','C','H','2','B','a','n','a','n','a','4','p','e','a','S','2','L' ,'b']
#y_pred=['0','n','N','0','n','3','L','b','S','P','N','n','R','c','W','2','B','A','n','A','n','A','4','P','e','A','S','2','L','b']
#y_pred=['A','B','B','i','e', '2', 'K', 'I', 'i', '0','m', 'a' ,'n' ,'a' ,'0' ,'3' ,'i','b','C' ,'U' ,'C' ,'U' ,'m' ,'b' ,'e' ,'R' ,'5']
print "The accuracy for "+argv[1]+" is "+str(accuracy(y_true,y_pred)*100)
 
