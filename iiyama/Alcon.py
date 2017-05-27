
# coding: utf-8

# In[2]:

get_ipython().magic('pylab inline')
import matplotlib.pyplot as plt


# In[3]:

import numpy as np
import cv2
from alcon_utils import AlconUtils


# In[4]:

alcon = AlconUtils("/share/alcon/dataset/")


# In[5]:

alcon.load_annotations_target("target_lv2.csv")
alcon.load_annotations_ground("groundtruth_lv2.csv")


# In[6]:

print (alcon.targets['1000'])
print (alcon.ground_truth['1000'])


# In[8]:

pylab.rcParams['figure.figsize'] = (20, 20)
rect = alcon.targets['1000'][1:5]
image = cv2.imread("/share/alcon/dataset/images/"+alcon.targets['1000'][0]+".jpg")
cv2.rectangle(image, rect[0:2], (rect[0]+rect[2],rect[1]+rect[3]),(255,0,0), 2)
plt.imshow(image[:,:,(2,1,0)])
print ("/share/alcon/dataset/images"+alcon.targets['1000'][0]+".jpg")


# In[17]:

tuple(rect[0]+rect[2],rect[1]+rect[3])


# In[ ]:



