
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


def loadDataset(filename):
    data = pd.read_csv(filename, header=None)
    return data


# In[3]:


def normilizeDataset(data):
    norm = np.sqrt(np.square(data.iloc[:,1:28]).sum(axis=1))
    newdata = data
    for row in range(0, data.shape[0]):
        for column in data.columns[1:]:
            newdata.iloc[row,column] = data.iloc[row,column] / norm[row]
    return newdata


# In[4]:


def main():
    file = 'HIGGS.csv'
    
    
    data = loadDataset(file)
    newdata = normilizeDataset(data)
    newdata.to_csv('newdata.csv', encoding='utf-8', index=False, header=False)


# In[ ]:


main()

