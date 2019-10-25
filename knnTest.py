# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 14:13:40 2019

@author: sosboy888
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt

img=cv2.imread('digits.png')
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cells=[np.hsplit(row,100) for row in np.vsplit(gray,50)]
x=np.array(cells)
train = x[:,:50].reshape(-1,400).astype(np.float32) # Size = (2500,400)
test = x[:,50:100].reshape(-1,400).astype(np.float32)
k=np.arange(10)
trainLabels=np.repeat(k,250)[:,np.newaxis]
testLabels=trainLabels.copy()
knn=cv2.KNearest()
knn.train(train,trainLabels)
ret,result,neighbours,dist=knn.find_nearest(test,k=5)
matches=result==testLabels
correct=np.count_nonzero(matches)
accuracy=correct*100.0/result.size
print accuracy
np.savez('knn_data.npz',train=train,train_labels=trainLabels)

