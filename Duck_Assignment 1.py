
# coding: utf-8

# In[76]:


import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB


# In[77]:


duck_image = []
duck_lable = []


        #cv2.imshow('image',dst)
       # cv2.waitKey(0)


# In[78]:


#正樣本
duck_location=os.listdir('duck_image/')
for duck in duck_location:
        imageName ="duck_image/"+duck
        img = cv2.imread(imageName,1)
        dst = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        info = dst.shape
        list_image_1 = []
        for i in range(info[0]):
            for j in range(info[1]):
                #pixel = float(dst[i][j])
                duck_image.append(img[i][j])
                duck_lable.append(0)
        
print(len(duck_image))
print(len(duck_lable))


# In[79]:


#負樣本
no_duck_image=os.listdir('no_duck_image/')
for no_duck in no_duck_image:
        imageName ="no_duck_image/"+no_duck
        img = cv2.imread(imageName,1)
        info = dst.shape
        list_image_2 = []
        for i in range(info[0]):
            for j in range(info[1]):
                duck_image.append(img[i][j])
                duck_lable.append(1)
print(len(duck_image))
print(len(duck_lable))


# In[80]:


data_train = np.array(duck_image)
data_lable = np.array(duck_lable)
data_train
#print(len(data_lable))


# In[81]:


clf_pf = GaussianNB()
clf_pf.fit(data_train,data_lable) 


# In[82]:


image1 = cv2.imread("full_duck.jpg",1)
imageInfo1 = image1.shape
height = imageInfo1[0]
width = imageInfo1[1]
deep=imageInfo1[2]
newInfo=(height,width,deep)
dst = np.zeros(newInfo,np.uint8)

for i in range(0,height):
    for j in range(0,width):
        dst[i,j] =image1[i,j]

for i in range(0,height):
    for j in range(0,width):
        (b,g,r)=image1[i,j]
        b=int(b)
        g=int(g)
        r=int(r)
        if clf_pf.predict([dst[i][j]])==0:
            dst[i,j] = (0,0,255)
            
cv2.imwrite("dst_full_duck.jpg",dst)
print("end")

