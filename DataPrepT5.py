#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import datetime as dt


# In[2]:


os.chdir("info")


# In[3]:


recId=os.listdir()


# In[4]:


lightsoff=[]
lightson=[]
recstart=[]
recstop=[]
for x in recId:
    os.chdir(x)
    os.chdir("info")
    f= open("lights_off_time.txt", "r")
    dd=dt.datetime.strptime("02/01/1900:"+f.readline().strip(),"%d/%m/%Y:%H:%M:%S")
    f.close()
    if dd.hour>12:
        dd=dd.replace(day=1)
    lightsoff.append(dd)
    f= open("lights_on_time.txt", "r")
    lightson.append(dt.datetime.strptime("02/01/1900:"+f.readline().strip(),"%d/%m/%Y:%H:%M:%S.%f"))
    f.close()
    f= open("rec_start_time.txt", "r")
    recstart.append(dt.datetime.strptime("01/01/1900:"+f.readline().strip(),"%d/%m/%Y:%H:%M:%S.%f"))
    f.close()
    f= open("rec_stop_time.txt", "r")
    recstop.append(dt.datetime.strptime("02/01/1900:"+f.readline().strip(),"%d/%m/%Y:%H:%M:%S"))
    f.close()
    os.chdir("../..")
os.chdir("..")


# In[5]:


import pyedflib


# In[6]:


os.chdir("SleepEDF")


# In[7]:


import numpy as np
import glob
Epochs = []
Labels = []
for i,x in enumerate(recId):
    psgFile=x+"-PSG.edf"
    hypnogramFile=glob.glob((psgFile[0:7]+'?-Hypnogram.edf'))
    psg = pyedflib.EdfReader(psgFile)
    hypnogram = pyedflib.EdfReader(hypnogramFile[0])
    Duration=hypnogram.readAnnotations()[1][:-1]
    if sum(Duration)*100!=len(psg.readSignal(0)):
        continue
    skipEND=(recstop[i]-lightson[i]).total_seconds()
    skipSTART=(lightsoff[i]-recstart[i]).total_seconds()
    if skipEND<0:
        noLights=psg.readSignal(0)[int(skipSTART*100):]
    else:
        noLights=psg.readSignal(0)[int(skipSTART*100):int(skipEND*-100)]
    Epoch=np.split(noLights,len(noLights)/3000)
    Label=[]
    for j in range(len(Epoch)):
        for g,x in enumerate(hypnogram.readAnnotations()[0]):
            if skipSTART+(j*30)<x:
                Label.append(hypnogram.readAnnotations()[2][g-1])
                break
    Epochs.extend(Epoch)
    Labels.extend(Label)
    psg._close()
    hypnogram._close()


# In[8]:


np.shape(Epochs)


# In[9]:


np.shape(Labels)


# In[10]:


delete=[]
for x in range(len(Epochs)):
    Labels[x]=Labels[x][-1]
    if Labels[x]=='?' or Labels[x]=='e':
        delete.append(x)
    else:
        if Labels[x]=='W':
            Labels[x]=0
        else:
            if Labels[x]=='4':
                Labels[x]=3
            else:
                if Labels[x]=='R':
                    Labels[x]=4
                else:
                    Labels[x]=int(Labels[x])


# In[ ]:





# In[11]:


np.shape(Labels)


# In[12]:


np.shape(Epochs)


# In[13]:


Epochs=np.delete(Epochs,delete,0)


# In[14]:


Labels=np.delete(Labels,delete,0)


# In[15]:



np.shape(Labels)


# In[16]:


np.shape(Epochs)


# In[17]:


np.unique(Labels)


# In[18]:


import torch


# In[19]:


Labels=Labels.astype(int)
Labels=torch.from_numpy(Labels)
Epochs=torch.from_numpy(Epochs)


# In[20]:


Epochs=Epochs.unsqueeze(1)


# In[21]:


Epochs.shape


# In[33]:


Epochs=Epochs.float()


# In[72]:


torch.save(Epochs, 'Epochs.pt') 


# In[73]:


torch.save(Labels, 'Labels.pt') 


# In[ ]:




