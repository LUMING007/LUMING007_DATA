
# coding: utf-8

# In[158]:


import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB


# In[182]:


duck_image = []
duck_lable = []

        #cv2.imshow('image',dst)
       # cv2.waitKey(0)


# In[183]:


#正樣本 100個
duck_location=os.listdir('duck_image/')
for duck in duck_location:
        imageName ="duck_image/"+duck
        img = cv2.imread(imageName,1)
        img = np.array(img)
        duck_image.append(img)
        duck_lable.append(0)       


# In[184]:


#負樣本 100個
no_duck_image=os.listdir('no_duck_image/')
for no_duck in no_duck_image:
        imageName ="no_duck_image/"+no_duck
        img = cv2.imread(imageName,1)
        img = np.array(img)
        duck_image.append(img)
        duck_lable.append(1)


# In[186]:


data_train = np.array(duck_image)
data_lable = np.array(duck_lable)


# In[187]:


clf_pf = GaussianNB()
clf_pf.fit(duck_image, duck_lable) 

