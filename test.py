import os,cv2
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split

from keras import backend as K
K.set_image_dim_ordering('tf')

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense,Dropout,Activation,Flatten
from keras.optimizers import SGD,RMSprop,adam
from keras.layers.convolutional import Conv2D,MaxPooling2D


from keras.preprocessing.image import ImageDataGenerator,img_to_array,load_img
from PIL import Image



img_rows=128
img_cols=128
num_channel=3

#resize the image to the fixe size

#define the number of classes
num_classes=5
#num_of_samples=img_data.shape[0]
#print("number of sampels input of label and classes:",num_of_samples)
#labels=np.ones((num_of_samples),dtype='int64')
#labels[0:399]=0
#labels[399:799]=1
#labels[799:1199]=2
#labels[1199:1599]=3
#labels[1599:1999]=4

names=['five hundered','one thousand','ten thousand','two thousand','five thousand']

#convert class label to on_hot encoding
#Y=np_utils.to_categorical(labels,num_classes)

#print("labels class and labels :", Y[0])

# input the new image for classfication
test_images=Image.open('/home/karoo/PycharmProjects/keras1/other/965.jpg')
test_images=test_images.resize((128,128),Image.ANTIALIAS)
test_images=img_to_array(test_images)
test_images=test_images.astype('float32')
test_images/=255
print("testing image input .....!")
print(test_images.shape)



if num_channel==1:
    if K.image_dim_ordering()=='th':
        test_images=np.expand_dims(test_images, axis=0)
        test_images = np.expand_dims(test_images, axis=0)
        print(test_images.shape)
    else:
        test_images=np.expand_dims(test_images,axis=3)
        test_images = np.expand_dims(test_images, axis=0)
        print(test_images.shape)
else:
    if K.image_dim_ordering()=='th':
        test_images=np.rollaxis(test_images,2,0)
        test_images=np.expand_dims(test_images,axis=0)
        print(test_images.shape)
    else:
        test_images=np.expand_dims(test_images,axis=0)
        print(test_images.shape)


#input_shape=img_data[0].shape
model=Sequential()

#layer1
model.add(Conv2D(32,(3,3),padding="same",input_shape=test_images[0].shape))
model.add(Activation('relu'))


#pooling layer
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))

#layer2
model.add(Conv2D(64,(3,3),padding="same" ))
model.add(Activation('relu'))

#pooling layer
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))


#layer3
model.add(Conv2D(128,(3,3) ,padding="same"))
model.add(Activation('relu'))


#FULLY connected
model.add(Flatten())
model.add(Dense(128))

model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(64))

model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(num_classes))
model.add(Activation('softmax'))




model.load_weights('/home/karoo/PycharmProjects/keras1/other/my_model.h5')
print((model.predict(test_images)))
print(model.predict_classes(test_images))

print("The end test Model")
