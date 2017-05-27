
# coding: utf-8

# In[4]:

get_ipython().magic('pylab inline')
import matplotlib.pyplot as plt

from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.models import Model,Sequential
from keras.layers import Dense, GlobalAveragePooling2D,Flatten,Dropout
from keras import backend as K
import keras


# In[5]:

from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1.0/255, shear_range=0.2, zoom_range=0.2, 
                                   horizontal_flip=False)
#test_datagen = ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_directory('/share/alcon/dataset/characters/',
                                                   target_size=(224,224),
                                                   batch_size=32,
                                                   class_mode='categorical')


# In[6]:

vgg16 = VGG16(weights='imagenet',include_top=False,input_shape=(224,224,3))


# In[7]:

vgg16layer = vgg16.get_layer('block5_pool')

n_model = Sequential()
n_model.add(Flatten(input_shape=vgg16layer.output_shape[1:]))
n_model.add(Dense(256,activation='relu'))
n_model.add(Dropout(0.5))
n_model.add(Dense(46,activation='sigmoid'))

model = Model(inputs=vgg16.input, outputs= n_model(vgg16layer.output) )

for layer in vgg16.layers:
    layer.trainable = False

#model.summary()
#n_model.summary()

model.compile(loss=keras.losses.binary_crossentropy,
              optimizer=keras.optimizers.Adadelta(),metrics=['accuracy'])


# In[8]:

history = model.fit_generator(train_generator, 
                              steps_per_epoch=100, # number of data / batchsize
                              epochs=1)


# In[9]:

model.save("mymodel.h5")


# In[27]:

from sklearn.externals import joblib
k = list(train_generator.class_indices.keys())
joblib.dump(k,"classes.pkl")


# In[28]:




# In[ ]:



