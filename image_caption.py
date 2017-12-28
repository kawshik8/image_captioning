
# coding: utf-8

# In[4]:


import gc
from keras.utils import multi_gpu_model
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.utils import np_utils, generic_utils
from keras.callbacks import EarlyStopping
from keras.layers.advanced_activations import PReLU, LeakyReLU
from keras.applications.vgg16 import preprocess_input
from keras.layers import Embedding,GRU,TimeDistributed,Dense,RepeatVector,Merge,LSTM,Bidirectional
from keras.preprocessing.text import one_hot
from keras.preprocessing import sequence
from keras.applications.vgg16 import VGG16
from keras.models import Model
#import MatplotLib
import h5py
import numpy as np
import cv2
import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import h5py
import numpy as np
import json


# In[19]:


batch_no = 0
gc.enable()
# In[2]:


filename = "/others/guest2/coc/coco_data.h5"

f = h5py.File(filename,'r')
print(list(f.keys())[0])   #images
image_group = list(f.keys())[0]
label_group = list(f.keys())[4]
label_len_group = list(f.keys())[2]
label_start_group = list(f.keys())[3]
label_end_group = list(f.keys())[1]

images = list(f[image_group])
label_len = list(f[label_len_group])
label_start = list(f[label_start_group])
label_end = list(f[label_end_group])
labels = list(f[label_group])
images = np.asarray(images)


# In[20]:


print("labels:5",labels[:5])


# In[21]:


def input_img_cap():

    while 1:
        
        batch_no = 0
        size2 = 16
        size1 = 0
        while size2<len(images):
            
            size1 = batch_no * 16

            size2 = (batch_no+1) * 16
            if size2 > len(images):
                size2 = 123287
            batch_size = label_end[size2-1]-label_start[size1]
            input_imgcap = {}
            input_imgcap['image'] = []#	np.zeros((batch_size,256,256,3))
            input_imgcap['caption_inp'] = []#np.zeros((batch_size,18))
            input_imgcap['caption_op'] = []#np.zeros((batch_size,18,9569))


            # In[ ]:


            #print(labels[0][5])


            # In[ ]:
            #size1 = batch_no * 32
            
            #size2 = (batch_no+1) * 32

            #if size2 > len(images):
            #    size2 = 123287

            for i,img in enumerate(images[size1:size2]):
                #print(i)
                j = label_start[i+size1] - 1
                img = np.array(img,dtype='float64')
                img = preprocess_input(img)
                while j < label_end[i+size1]:
                    
                    input_imgcap['image'].append(img)
                    labels[j] = list(labels[j])
                    #print(len(labels[j]),labels[j])
                    labels[j].insert(0,9568)  ##inserting start token for each caption

                    # for itt,labelk in enumerate(labels[j]):
                    #     if labelk!=0:
                    #         continue
                    #     else:
                    #         labels[j].insert(itt,9569)

                    labels[j].insert(17,9569)  ##inserting end token for each caption
                    #print(len(labels[j]),labels[j])
                    #labels[j] = np.array(img,dtype='float64')
                    #print(labels[j])
                    input_imgcap['caption_inp'].append(np.array(labels[j],dtype='float64'))
                    caption = np.zeros((9569))
                    #labels[j] = list(labels[j])
                    for label in labels[j]:
                        if label!=0:
                            caption[label-1] = 1
                    #print(caption)
                    caption = np.array(caption,dtype = 'float64')
                    input_imgcap['caption_op'].append(caption)
                    #print(caption)
                    j+=1
#            if batch_no%10==0:
 #               gc.collect()
            batch_no+=1        
            input_imgcap['image'] = np.array(input_imgcap['image'])
	    input_imgcap['caption_inp'] = np.array(input_imgcap['caption_inp'])
            input_imgcap['caption_op'] = np.array(input_imgcap['caption_op'])
	    #print(input_imgcap['image'].shape,input_imgcap['caption_inp'].shape,input_imgcap['caption_op'].shape)
            
            yield [np.array(input_imgcap['image']),np.array(input_imgcap['caption_inp'])],np.array(input_imgcap['caption_op'])





# In[8]:


def vggmodel():
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(256,256,3)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    #model.add(Dense(1000, activation='softmax'))
    return model


# In[9]:


max_caption_len = 18
vocab_size = 9569


# In[ ]:


#with tf.device('/gpu:0'):  
print "VGG loading"
base_model = vggmodel()
#base_model.layers.pop()
base_model.trainable = False

#print(base_model.summary())
print "VGG loaded"
# let's load the weights from a save file.
# image_model.load_weights('weight_file.h5')

# next, let's define a RNN model that encodes sequences of words
# into sequences of 128-dimensional word vectors.
print "Text model loading"
language_model = Sequential()
language_model.add(Embedding(vocab_size, 256, input_length=max_caption_len))
language_model.add(GRU(units=128, return_sequences=True))
language_model.add(TimeDistributed(Dense(128)))
print(language_model.summary())
print "Text model loaded"
# let's repeat the image vector to turn it into a sequence.
# print "Repeat model loading"
base_model.add(RepeatVector(max_caption_len))
# print "Repeat model loaded"
# the output of both models will be tensors of shape (samples, max_caption_len, 128).
# let's concatenate these 2 vector sequences.
print "Merging"
model = Sequential()
model.add(Merge([base_model, language_model], mode='concat', concat_axis=-1))
# let's encode this vector sequence into a single vector
model.add(GRU(256, return_sequences=False))
# which will be used to compute a probability
# distribution over what the next word in the caption should be!
model.add(Dense(vocab_size))
model.add(Activation('softmax'))
print(model.summary())
model1 = multi_gpu_model(model,gpus=2)
model1.compile(loss='categorical_crossentropy', optimizer='rmsprop',metrics = ['accuracy'])
print "Merged"

print "Data preprocessing done"


# In[22]:


history = model1.fit_generator(input_img_cap(), epochs=25, steps_per_epoch = 7706, verbose = 1)

model1.save('/others/guest2/coc/model.weight.end.hdf5')


# In[ ]:


print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
#plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('/others/guest2/coc/accuracy_model.png')
# summarize history for loss
plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('/others/guest2/coc/loss_model.png')




