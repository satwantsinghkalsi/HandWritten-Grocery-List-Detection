# Import the modules
import cv2
from sklearn.externals import joblib
from skimage.feature import hog
import numpy as np
import argparse
from sklearn.svm import LinearSVC
from collections import Counter

import os
import fnmatch

# Load the classifier
clf = joblib.load("chars_cls.pkl")

list_hog_fd = []
responses = []
# Read the input image
for dirpath, dirs, files in os.walk('/home/vaishthiru/Downloads/FinalProject/tst'):
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
	    # construct the list of bounding boxes and sort them from top to  bottom
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
                    nbr = clf.predict(np.array([roi_hog_fd], 'float32'))
                    #cv2.imshow('im',im)
                    #cv2.waitKey(0)
                    #print (chr(nbr)) 
                    result.append(chr(nbr))
        for nbrs in result:
            print(nbrs, end="") 
        cv2.imshow('im',im)
        cv2.waitKey(0)
        print()
