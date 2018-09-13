
# coding: utf-8

# In[1]:


from vgg_17_keras import *
import numpy as np
import cv2
import matplotlib.pyplot as plt

from oxford_classes import classes
import os, cv2


# In[2]:


def get_data(pics_folder_path, num_classes):
    X, Y = [],[]
    names = os.listdir(pics_folder_path)
    for i,n in enumerate(names):
        im = cv2.imread(pics_folder_path+n)
        resized_im = cv2.resize(im, (224, 224)).astype(np.float32)
        resized_im = (resized_im - np.min(resized_im))/(np.max(resized_im)-np.min(resized_im))
        resized_im = resized_im.transpose((2,0,1))

        X.append(resized_im)

        y_ = np.zeros(num_classes)
        im_class = n[:n.rfind('_')]
        index = 17
        if im_class in classes:
            index = classes.index(im_class)
        y_[index] = 1
        Y.append(y_)

        #delete this!!
        # if i == 9:
        #     break
    
    X = np.stack(X, axis=0)
    Y = np.stack(Y, axis=0)
    return X,Y
    
    
    


# In[3]:


path = 'oxbuild_images/'
num_classes = 18

X,Y = get_data(path, num_classes)
X.shape, Y.shape


# In[4]:


# divide data
num_examples = int(X.shape[0])
num_tests = int(0.8*num_examples)
mask = np.full(X.shape[0], False)
mask[:num_tests] = True
np.random.shuffle(mask)

X_train = X[mask]
Y_train = Y[mask]
X_test = X[np.invert(mask)]
Y_test = Y[np.invert(mask)]


# In[5]:


model=base_model()


# In[6]:


# model.predict(X_train[0])
# im = X_train
# im = im.transpose((2,0,1))
# im = np.expand_dims(im, axis=0)
# y_ = Y_train
# print(y_.shape)
# im.shape, y_.shape
# y_ = model.predict(im)
# print(im.shape, y_.shape)
# model.fit(im, y_, epochs=1)
# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())



# model = train_model(model, X_train, Y_train)
# y_test = model.predict(X_test)
# print(np.count_nonzero(y_test==Y_test)/X_test.shape[0])
#
# save_weights(model)
# print("end")


# In[ ]:

model = train_model(model, X_train, Y_train)
print('finish')
save_weights(model)
# y = model.predict(X_test)
# print(np.count_nonzero(y==Y_test)/Y_test.shape[0])
