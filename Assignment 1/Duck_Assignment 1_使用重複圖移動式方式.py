
# coding: utf-8

# In[2]:


import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB


# In[3]:


duck_image = [] 
duck_lable = []


# In[4]:


#正樣本 輸入
duck_location=os.listdir('duck_image/')
for duck in duck_location:
        imageName ="duck_image/"+duck
        img = cv2.imread(imageName,1)
        dst = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        info = dst.shape
        list_image_1 = []
        for i in range(info[0]):
            for j in range(info[1]):
                duck_image.append(img[i][j])
                duck_lable.append(0)
        
print(len(duck_image))
print(len(duck_lable))


# In[5]:


#負樣本 輸入
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


# In[6]:


# 轉換陣列
data_train = np.array(duck_image)
data_lable = np.array(duck_lable)


# In[7]:


#高斯貝葉斯分類器 gaussianNB 來訓練樣本
clf_pf = GaussianNB()
clf_pf.fit(data_train,data_lable) 


# In[31]:


#測試 並將 鴨子用綠色圓圈標記

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
Total_duck=0       #設置鴨子總數計數參數
# 設置 偵測鴨子的視窗大小
D_dst_height =15
D_dst_width =15
D_dst =np.zeros([D_dst_height,D_dst_width],np.uint8)
# 設置 P 像素比例
P = int(0.88*D_dst_height*D_dst_width)
print(P)

for i in range(0,height,8):
    for j in range(0,width,8):
        D_dst=dst[i:i+D_dst_height,j:j+D_dst_width]
        count =0
        D_dst.shape
        #設置畫圓的中心點
        x=int(D_dst.shape[0]/2)
        y=int(D_dst.shape[1]/2)
        #設置使用clf_pf.predict函數來偵測鴨子像素點.若在n*n中鴨子像素點佔一定比例.視為鴨子存在
        for h in range(0,D_dst.shape[0]):
            for w in range(0,D_dst.shape[1]):
                if clf_pf.predict([D_dst[h][w]])==0:
                    count=count+1           
        if count>P:
            Total_duck=Total_duck+1
            cv2.circle(D_dst,(x,y),5,(0,255,0),2)
            #cv2.imshow('dst',D_dst)
            #cv2.imshow('dst',dst)
            #cv2.waitKey(0)

cv2.imwrite("dst_full_duck.jpg",dst)  
print('Total_duck',Total_duck)

