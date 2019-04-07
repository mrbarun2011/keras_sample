
# coding: utf-8

# In[341]:


import keras
from keras import backend as K
from keras.models import Sequential 
from keras.layers import Activation, Dropout
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy


# In[342]:


import numpy as np
from keras.layers.core import Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[343]:


gen = ImageDataGenerator(rotation_range=10,width_shift_range=0.1,height_shift_range=0.1,shear_range=0.15,zoom_range=0.1,channel_shift_range=10.,horizontal_flip=True)


# In[344]:


train_path='D:\\jupyter\\tarun\\train'
valid_path='D:\\jupyter\\tarun\\valid'
test_path='D:\\jupyter\\tarun\\test'


# In[345]:


train_batches=ImageDataGenerator().flow_from_directory(directory=train_path, target_size=(224,224),classes=['dog','cat'], batch_size=10, )
valid_batches=ImageDataGenerator().flow_from_directory(directory=valid_path, target_size=(224,224),classes=['dog','cat'],batch_size=10,)
test_batches=ImageDataGenerator().flow_from_directory(directory=test_path, target_size=(224,224),classes=['dog','cat'], batch_size=10,)


# In[346]:


from keras.layers import MaxPooling2D


# In[347]:


model = Sequential([

   Conv2D(32,(3, 3), activation="relu",input_shape= (224,224,3)),
   Flatten(),
   Dense(2, activation='sigmoid'),
    
 ])


# In[348]:


model.compile(Adam(lr=.0001),loss='categorical_crossentropy', metrics=['accuracy'])


# In[349]:


model.fit_generator(train_batches, steps_per_epoch=10, validation_data=valid_batches, validation_steps=5,epochs=5,verbose=2)


# In[350]:


#predict


# In[351]:


test_imgs, test_labels = next(test_batches)
#plots(test_imgs,title=test_labels)


# In[352]:


#test_labels = test_labels[:,0]
test_labels[:,0]


# In[353]:


predictions = model.predict_generator(test_batches, steps=1,verbose = 0)


# In[354]:


predictions[:,0]


# In[355]:


#cm = confusion_matrix(test_labels, predictions[:,0])


# In[356]:


#confusion matrix
import seaborn as sns


# In[357]:


test_labels


# In[358]:


from sklearn.metrics import classification_report


# In[359]:


print(classification_report(test_labels,predictions))


# In[360]:


sns.heatmap(confusion_matrix(test_labels[:,0],predictions[:,0]),annot=True,fmt='.5g') 


# In[361]:


#build fine tuned VGG16 model


# In[362]:


from keras.applications.vgg16 import VGG16


# In[363]:


vgg16_model = VGG16


# In[364]:


vgg16_model = keras.applications.vgg16.VGG16()


# In[365]:


vgg16_model.summary()


# In[366]:


type(vgg16_model)


# In[367]:


model= Sequential()
for layer in vgg16_model.layers:
    model.add(layer)


# In[368]:


model.summary()


# In[369]:


model.layers.pop()


# In[370]:


model.summary()


# In[371]:


##frezzing the layers
for layer in model.layers:
    layer.trainable=False


# In[372]:


model.add(Dense(2,activation='softmax'))


# In[373]:


model.summary()


# In[ ]:





# In[374]:


#Train the fine-tuned VGG16 model


# In[375]:


model.compile(Adam(lr=.0001), loss='categorical_crossentropy',metrics=['accuracy'])


# In[376]:


model.fit_generator(train_batches, steps_per_epoch=10,validation_data=valid_batches,validation_steps=10,epochs=5,verbose=2)


# In[385]:


#predict using fine-tuned vgg16 model


# In[386]:


#test_imgs, test_labels = next(test_batches)
#plots(test_imgs, title=test_labels)


# In[387]:


test_labels[:,0]


# In[388]:


predictions = model.predict_generator(test_batches, steps= 1, verbose=0)


# In[389]:


k=np.round(predictions[:,0])


# In[390]:


k


# In[391]:



sns.heatmap(confusion_matrix(test_labels[:,0],k),annot=True,fmt='.5g') 


# In[392]:


from sklearn.metrics import classification_report
print(classification_report(test_labels[:,0],k))


# In[ ]:




