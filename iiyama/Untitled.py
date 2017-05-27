
# coding: utf-8

# In[3]:

get_ipython().magic('pylab inline')
import matplotlib.pyplot as plt
import cv2
import keras


# In[4]:

model = keras.models.load_model("mymodel.h5")


# In[7]:

image = cv2.imread("/share/alcon/dataset/characters/U+3042/U+3042_200021853-00007_1_X0605_Y1607.jpg")


# In[12]:

x = cv2.resize(image,(224,224)) / 255.0
x = x.reshape((1,x.shape))


# In[ ]:



