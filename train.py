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

PATH=os.getcwd()
#definde data pach
data_path='enter you path'
data_dir_list=os.listdir(data_path)

img_rows=128
img_cols=128
num_channel=3

#resize the image to the fixe size






img_data_list=[]

for dataset in data_dir_list:
    img_list=os.listdir(data_path+'/'+dataset)
    print('loaded the images of dataset'+'{}\n'.format(dataset))
    for img in img_list:
        input_img=Image.open(data_path+'/'+dataset+'/'+img)
        imResize=input_img.resize((128,128),Image.ANTIALIAS)
        img1_data=img_to_array(imResize)
        img_data_list.append(img1_data)



img_data=np.array(img_data_list)
img_data=img_data.astype('float32')
print('image data size save in matrix :',np.size(img_data))
img_data/=255

print(img_data.shape)


if num_channel==1:
    if K.image_dim_ordering()=='th':
        img_data=np.expand_dims(img_data,axis=1)
        print(img_data.shape)
    else:
        img_data=np.expand_dims(img_data,axis=4)
        print(img_data.shape)
else:
    if K.image_dim_ordering()=='th':
        img_data=np.rollaxis(img_data,3,1)
        print(img_data.shape)
    else:
        img_data = np.rollaxis(img_data,3,4)
        print(img_data.shape)





#define the number of classes
num_classes=5
num_of_samples=img_data.shape[0]
print("number of sampels input of label and classes:",num_of_samples)
labels=np.ones((num_of_samples),dtype='int64')
labels[0:399]=0
labels[399:799]=1
labels[799:1199]=2
labels[1199:1599]=3
labels[1599:1999]=4

names=['five hundered','one thousand','ten thousand','two thousand','five thousand']

#convert class label to on_hot encoding
Y=np_utils.to_categorical(labels,num_classes)

#print("labels class and labels :", Y[0])



cv2.destroyAllWindows()
#created image data test and trainnig
x,y=shuffle(img_data,Y,random_state=2)
#print(x[0],y[0])

X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=4)
#print(X_train[0])
print(y_train)







#building and training CNN
input_shape=img_data[0].shape

model=Sequential()

#layer1
model.add(Conv2D(32,(3,3),padding="same",input_shape=input_shape))
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



#model.compile(loss='catgorical_crossentropy',optimizer='adadelte',metrics=['accuracy'])
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#Train
num_epoch=200
hist=model.fit(X_train,y_train,batch_size=32,nb_epoch=num_epoch,verbose=1,validation_split=0.2)

model.save('./my_model.h5')
#%%
#validation loss and acurrecy
train_loss=hist.history['loss']
val_loss=hist.history['val_loss']
train_acc=hist.history['acc']
val_acc=hist.history['val_acc']

xc=range(num_epoch)


plt.figure(1,figsize=(7,5))
plt.plot(xc,train_loss)
plt.plot(xc,val_loss)
plt.xlabel('number of epochs')
plt.ylabel('loss')
plt.title('train loss vs val_loss')
plt.grid(True)
plt.legend(['train','val'])

#plt.style.use(['classic'])

plt.figure(2,figsize=(7,5))
plt.plot(xc,train_acc)
plt.plot(xc,val_acc)
plt.xlabel('num of epochs')
plt.ylabel('accurecy')
plt.title('train_acc vs val_acc')
plt.grid(True)
plt.legend(['train','val'],loc=5)

plt.show()


# test
score= model.evaluate(X_test,y_test )


test_image=X_test
#print(test_image.shape)
print(model.predict(test_image))
print(model.predict_classes(test_image))
print(y_test)
print('Test loss:',score[0])
print('TEST acurracy:',score[1])


# input the new image for classfication
test_images=Image.open('/content/drive/My Drive/app/50000.jpg')
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


print((model.predict(test_images)))
print(model.predict_classes(test_images))

print("The end test Model")




#creat plot from layer
def get_featuremaps(model,layer_idx,x_batch):
    get_activations=K.function([model.layers[0].input,K.learning_phase()],[model.layers[layer_idx].output,])
    activations=get_activations([x_batch,0])
    return activations




layer_num=1
filter_num=30


activations=get_featuremaps(model,int(layer_num),test_image)

print(np.shape(activations))
feature_maps=activations[0][0]
#print(np.shape(feature_maps))


print(np.shape(feature_maps))




#fig=plt.figure(figsize=(16,16))

#plt.imshow(feature_maps[:,:,filter_num],cmap='gray')

#plt.savefig("featuremaps-layer-{}".format(layer_num)+"-fiternum-{}".format(filter_num)+'.jpg')

#plt.show()


num_of_featuremaps=feature_maps.shape[2]
fig=plt.figure(figsize=(16,16))
plt.title("/content/drive/My Drive/app/featuremaps-layer-{}".format(layer_num))
subplot_num=int(np.ceil(np.sqrt(num_of_featuremaps)))


for i in range(int(num_of_featuremaps)):
    ax=fig.add_subplot(subplot_num,subplot_num,i+1)
#Colormap Accent, Accent_r, Blues, Blues_r, BrBG, BrBG_r, BuGn, BuGn_r, BuPu, BuPu_r, CMRmap, CMRmap_r, Dark2, Dark2_r, GnBu, GnBu_r, Greens, Greens_r, Greys, Greys_r, OrRd, OrRd_r, Oranges, Oranges_r, PRGn, PRGn_r, Paired, Paired_r, Pastel1, Pastel1_r, Pastel2, Pastel2_r, PiYG, PiYG_r, PuBu, PuBuGn, PuBuGn_r, PuBu_r, PuOr, PuOr_r, PuRd, PuRd_r, Purples, Purples_r, RdBu, RdBu_r, RdGy, RdGy_r, RdPu, RdPu_r, RdYlBu, RdYlBu_r, RdYlGn, RdYlGn_r, Reds, Reds_r, Set1, Set1_r, Set2, Set2_r, Set3, Set3_r, Spectral, Spectral_r, Wistia, Wistia_r, YlGn, YlGnBu, YlGnBu_r, YlGn_r, YlOrBr, YlOrBr_r, YlOrRd, YlOrRd_r, afmhot, afmhot_r, autumn, autumn_r, binary, binary_r, bone, bone_r, brg, brg_r, bwr, bwr_r, cividis, cividis_r, cool, cool_r, coolwarm, coolwarm_r, copper, copper_r, cubehelix, cubehelix_r, flag, flag_r, gist_earth, gist_earth_r, gist_gray, gist_gray_r, gist_heat, gist_heat_r, gist_ncar, gist_ncar_r, gist_rainbow, gist_rainbow_r, gist_stern, gist_stern_r, gist_yarg, gist_yarg_r, gnuplot, gnuplot2, gnuplot2_r, gnuplot_r, gray, gray_r, hot, hot_r, hsv, hsv_r, inferno, inferno_r, jet, jet_r, magma, magma_r, nipy_spectral, nipy_spectral_r, ocean, ocean_r, pink, pink_r, plasma, plasma_r, prism, prism_r, rainbow, rainbow_r, seismic, seismic_r, spring, spring_r, summer, summer_r, tab10, tab10_r, tab20, tab20_r, tab20b, tab20b_r, tab20c, tab20c_r, terrain, terrain_r, viridis, viridis_r, winter, winter_r
    ax.imshow(feature_maps[:,:,i],cmap='BrBG')
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()


plt.show()
fig.savefig("/content/drive/My Drive/app/featurmaps-layer-{}".format(layer_num)+'.jpg')



# created confusion matrix for result
from sklearn.metrics import classification_report,confusion_matrix
import itertools

y_pred=model.predict(X_test)
print(y_pred)
y_pred=np.argmax(y_pred,axis=1)
print(y_pred)

target_names=['class0(five hundered)','class1(one thousand)','class2(ten thousand)','class3(two thousand)','class4(five thousand)']

print(classification_report(np.argmax(y_test,axis=1),y_pred,target_names=target_names))
cnf_matrix=confusion_matrix(np.argmax(y_test,axis=1),y_pred)
print(confusion_matrix(np.argmax(y_test,axis=1),y_pred))


#define confusion matrix
def plot_confusion_matrix(cm,classes,normalize=False,title='confusion matrix',cmap=plt.cm.Blues):
    plt.imshow(cm,interpolation='nearest',cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks=np.arange(len(classes))
    plt.xticks(tick_marks,classes,rotation=45)
    plt.yticks(tick_marks, classes)
    plt.show()

    if normalize:
        cm=cm.astype('float')/cm.sum(axis=1)[:,np.newaxis]
        print("normalized confuison matrix")
    else:
        print("confuion matrix withowt normalize")
    print(cm)

    tresh=cm.max()/2
    for i,j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):
        plt.text(j,i,cm[i,j])

    plt.tight_layout()




plot_confusion_matrix(cnf_matrix,classes=target_names,title='confuion matrix')
