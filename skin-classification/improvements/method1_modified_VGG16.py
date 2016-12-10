'''method 1: training a small network from scratch'''
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import random
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.regularizers import WeightRegularizer, ActivityRegularizer
from keras.layers.normalization import BatchNormalization
from keras.utils.visualize_util import plot
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')

# Options
SAVE_WEIGHTS = 1
PRINT_MODEL = 0

time_elapsed = 0
random.seed(333)

# dimensions of our images.
#img_width, img_height = 224, 224
img_width, img_height = 256, 256

# Paths to set
model_name = "method1_VGG16"
model_path = "models_trained/" +model_name+"/"
weights_path = "models_trained/"+model_name+"/weights/"
train_data_dir = '/imatge/aromero/work/image-classification/isbi-dataset/train'
validation_data_dir = '/imatge/aromero/work/image-classification/isbi-dataset/test'

# Network Parameters
nb_train_samples = 896
nb_validation_samples = 378
batch_size = 32
nb_epoch = 20

# Create directories for the models
if not os.path.exists(model_path):
	os.makedirs(model_path)
	os.makedirs(weights_path)

# Initialize result files
f_train = open(model_path+model_name+"_scores_training.txt", 'w')
f_test = open(model_path+model_name+"_scores_test.txt", 'w')
f_scores = open(model_path+model_name+"_scores.txt", 'w')

print('-'*30)
print('Defining modified VGG16 architecture...')
print('-'*30)
# build the modified VGG16 network
model = Sequential()

#conv-spatial batch norm - relu #1
model.add(ZeroPadding2D((2,2),input_shape=(3,img_width, img_height)))
model.add(Convolution2D(64,5,5,subsample=(2,2),W_regularizer=WeightRegularizer(l1=1e-7,l2=1e-7)))
model.add(BatchNormalization(epsilon=1e-06, mode=0, axis=1, momentum=0.9))
model.add(Activation('relu'))
print "added conv1"

#conv-spatial batch norm - relu #2
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(64,3,3,subsample=(1,1)))
model.add(BatchNormalization(epsilon=1e-06, mode=0, axis=1, momentum=0.9))
model.add(Activation('relu'))
print "added conv2"

#conv-spatial batch norm - relu #3
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128,3,3,subsample=(2,2)))
model.add(BatchNormalization(epsilon=1e-06, mode=0, axis=1, momentum=0.9))
model.add(Activation('relu'))
model.add(Dropout(0.5))
print "added conv3"

#conv-spatial batch norm - relu #4
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128,3,3,subsample=(1,1)))
model.add(BatchNormalization(epsilon=1e-06, mode=0, axis=1, momentum=0.9))
model.add(Activation('relu'))
print "added conv4"

#conv-spatial batch norm - relu #5
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256,3,3,subsample=(2,2)))
model.add(BatchNormalization(epsilon=1e-06, mode=0, axis=1, momentum=0.9))
model.add(Activation('relu'))
print "added conv5"

#conv-spatial batch norm - relu #6
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256,3,3,subsample=(1,1)))
model.add(BatchNormalization(epsilon=1e-06, mode=0, axis=1, momentum=0.9))
model.add(Activation('relu'))
model.add(Dropout(0.25))
print "added conv6"

#conv-spatial batch norm - relu #7
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512,3,3,subsample=(2,2)))
model.add(BatchNormalization(epsilon=1e-06, mode=0, axis=1, momentum=0.9))
model.add(Activation('relu'))
print "added conv7"

#conv-spatial batch norm - relu #8
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512,3,3,subsample=(1,1)))
model.add(BatchNormalization(epsilon=1e-06, mode=0, axis=1, momentum=0.9))
model.add(Activation('relu'))
print "added conv8"


#conv-spatial batch norm - relu #9
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(1024,3,3,subsample=(2,2)))
model.add(BatchNormalization(epsilon=1e-06, mode=0, axis=1, momentum=0.9))
model.add(Activation('relu'))
print "added conv9"
model.add(Dropout(0.25))

#Affine-spatial batch norm -relu #10
model.add(Flatten())
model.add(Dense(512,W_regularizer=WeightRegularizer(l1=1e-5,l2=1e-5)))
model.add(BatchNormalization(epsilon=1e-06, mode=0, axis=1, momentum=0.9))
model.add(Activation('relu'))
print "added affine!"
model.add(Dropout(0.5))

#affine layer w/ softmax activation added
model.add(Dense(1,activation='softmax',W_regularizer=WeightRegularizer(l1=1e-5,l2=1e-5)))#pretrained weights assume only 100 outputs, we need to train this layer from scratch
print "added final affine"

model.summary()

if(PRINT_MODEL):
    print('-'*30)
    print('Printing model...')
    print('-'*30)
    plot(model, to_file='method1_VGG16_model.png')

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
                    rescale=1./255,
                    shear_range=0,
                    rotation_range=40, # randomly rotate images in the range (degrees, 0 to 180)
                    width_shift_range=0.2, # randomly shift images horizontally (fraction of total width)
                    height_shift_range=0.2, # randomly shift images vertically (fraction of total height)
                    zoom_range=0.2,
                    horizontal_flip=True, # randomly flip images
                    vertical_flip=False)  # randomly flip images

print('-'*30)
print('Data augmentation...')
print('-'*30)
# this is the augmentation configuration we will use for testing:
test_datagen = ImageDataGenerator(rescale=1./255)

print('-'*30)
print('Creating train batches...')
print('-'*30)
train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary')

print('-'*30)
print('Creating test batches...')
print('-'*30)
validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary')

print('-'*30)
print('Training model...')
print('-'*30)
for epoch in range(1,nb_epoch+1):

    t0 = time.time()
    print ("Number of epoch: " +str(epoch)+"/"+str(nb_epoch))

    scores = model.fit_generator(
                train_generator,
                samples_per_epoch=nb_train_samples,
                nb_epoch=1,
                validation_data=validation_generator,
                nb_val_samples=nb_validation_samples)
    time_elapsed = time_elapsed + time.time() - t0
    print ("Time Elapsed: " +str(time_elapsed))

    score_train = model.evaluate_generator(generator=train_generator, val_samples=nb_train_samples, max_q_size=1)
    f_train.write(str(score_train)+"\n")

    score_test = model.evaluate_generator(generator=validation_generator, val_samples=nb_validation_samples, max_q_size=1)
    f_test.write(str(score_test)+"\n")

    f_scores.write(str(score_train[0])+","+str(score_train[1])+","+str(score_test[0])+","+str(score_test[1])+"\n")

if(SAVE_WEIGHTS):
	print('-'*30)
	print('Saving weights...')
	print('-'*30)
	model.save_weights(weights_path+model_name+"_weights.h5")

print('-'*30)
print('Model evaluation...')
print('-'*30)
score_train = model.evaluate_generator(generator=train_generator, val_samples=nb_train_samples, max_q_size=1)
print('Train Loss:', score_train[0])
print('Train Accuracy:', score_train[1])

score_test = model.evaluate_generator(generator=validation_generator, val_samples=nb_validation_samples, max_q_size=1)
print('Test Loss:', score_test[0])
print('Test Accuracy:', score_test[1])

f_train.close()
f_test.close()
f_scores.close()

# Save time elapsed
f = open(model_path+model_name+"_time_elapsed.txt", 'w')
f.write(str(time_elapsed))
f.close()
