
# coding: utf-8

# In[1]:


def distance(s1, s2):
    m=len(s1)+1
    n=len(s2)+1

    tbl = {}
    for i in range(m): tbl[i,0]=i
    for j in range(n): tbl[0,j]=j
    for i in range(1, m):
        for j in range(1, n):
            cost = 0 if s1[i-1] == s2[j-1] else 1
            tbl[i,j] = min(tbl[i, j-1]+1, tbl[i-1, j]+1, tbl[i-1, j-1]+cost)

    return tbl[i,j]


# In[7]:


inputs = input("請輸入一個字串: ")
string = inputs.strip()
file = open("dictionary.txt",'r')
all_line = file.readlines()
decide=1
for line in all_line:
     if distance(string,line.strip())==0:
            print("輸入正確",string)
            print("字典中有:",line)
            decide=0
if decide==1:
    for line in all_line:
        if distance(string,line.strip())<3:
            print("相似字: ",line.strip()," 距離: ",distance(string,line.strip()))

