import os
import h5py
import numpy as np
import random
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils.visualize_util import plot
from keras import backend as K
K.set_image_dim_ordering('th')

# Options
SAVE_WEIGHTS = 0
PRINT_MODEL = 0

time_elapsed = 0
random.seed(333)

# Paths to set
#model_name = "method3_VGG16_batch_size_256"
model_name = "method3_VGG16"
model_path = "models_trained/" +model_name+"/"
saving_weights_path = "models_trained/"+model_name+"/weights/"
weights_path = '/imatge/aromero/work/image-classification/weights/vgg16_weights.h5'
top_model_weights_path = '/imatge/aromero/work/image-classification/MIDDLE_Skin_Classification/models_trained/method1_VGG16/weights/method1_skin_weights_20epochs.h5'
train_data_dir = '/imatge/aromero/work/image-classification/isbi-dataset/train'
validation_data_dir = '/imatge/aromero/work/image-classification/isbi-dataset/test'

# Network Parameters
img_width, img_height = 512, 384   # Dimensions of our images
nb_train_samples = 896
nb_validation_samples = 378
nb_epoch = 30
batch_size = 32
dropout = 0.8
freeze = 18
print('Dropout: '+str(dropout))
print('Freezing: '+str(freeze)+' layers')


# Create directories for the models
if not os.path.exists(model_path):
        os.makedirs(model_path)
        os.makedirs(weights_path)

# Initialize result files
f_train = open(model_path+model_name+"_scores_training.txt", 'w')
f_test = open(model_path+model_name+"_scores_test.txt", 'w')
f_scores = open(model_path+model_name+"_scores.txt", 'w')

# build the VGG16 network
model = Sequential()
model.add(ZeroPadding2D((1, 1), input_shape=(3, img_width, img_height)))

model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

# load the weights of the VGG16 networks
# (trained on ImageNet, won the ILSVRC competition in 2014)
# note: when there is a complete match between your model definition
# and your weight savefile, you can simply call model.load_weights(filename)
print('-'*30)
print('Loading weights...')
print('-'*30)
assert os.path.exists(weights_path), 'Model weights not found (see "weights_path" variable in script).'
f = h5py.File(weights_path)
for k in range(f.attrs['nb_layers']):
    if k >= len(model.layers):
        # we don't look at the last (fully-connected) layers in the savefile
        break
    g = f['layer_{}'.format(k)]
    weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
    model.layers[k].set_weights(weights)
f.close()
print('Model loaded.')

# build a classifier model to put on top of the convolutional model
print('-'*30)
print('Building a classifier model on top of the ConvNet...')
print('-'*30)
top_model = Sequential()
top_model.add(Flatten(input_shape=model.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(dropout))
top_model.add(Dense(1, activation='sigmoid'))

# note that it is necessary to start with a fully-trained
# classifier, including the top classifier,
# in order to successfully do fine-tuning
top_model.load_weights(top_model_weights_path)

# add the model on top of the convolutional base
model.add(top_model)
print('-'*30)
print('Defining the final model...')
print('-'*30)
model.summary()

if(PRINT_MODEL):
    print('-'*30)
    print('Printing model...')
    print('-'*30)
    plot(model, to_file='method3_VGG_model.png')

# set the first 25 layers (up to the last conv block)
# to non-trainable (weights will not be updated)
for layer in model.layers[:freeze]:
    layer.trainable = False

# compile the model with a SGD/momentum optimizer
# and a very slow learning rate.
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])

# prepare data augmentation configuration
print('-'*30)
print('Data augmentation...')
print('-'*30)
train_datagen = ImageDataGenerator(
                    rescale=1./255,
                    shear_range=0,
                    rotation_range=40, # randomly rotate images in the range (degrees, 0 to 180)
                    width_shift_range=0.2, # randomly shift images horizontally (fraction of total width)
                    height_shift_range=0.2, # randomly shift images vertically (fraction of total height)
                    zoom_range=0.2,
                    horizontal_flip=True, # randomly flip images
                    vertical_flip=False)  # randomly flip images

test_datagen = ImageDataGenerator(rescale=1./255)

print('-'*30)
print('Creating batches...')
print('-'*30)
train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary')

# fine-tune the model
print('-'*30)
print('Fine-tunning the model...')
print('-'*30)
for epoch in range(1,nb_epoch+1):

 scores = model.fit_generator(
                train_generator,
                samples_per_epoch=nb_train_samples,
                nb_epoch=1,
                validation_data=validation_generator,
                nb_val_samples=nb_validation_samples)

    score_train = model.evaluate_generator(generator=train_generator, val_samples=nb_train_samples, max_q_size=1)
    f_train.write(str(score_train)+"\n")

    score_test = model.evaluate_generator(generator=validation_generator, val_samples=nb_validation_samples, max_q_size=1)
    f_test.write(str(score_test)+"\n")

    f_scores.write(str(score_train[0])+","+str(score_train[1])+","+str(score_test[0])+","+str(score_test[1])+1])+"\n")

if(SAVE_WEIGHTS):
    print('-'*30)
    print('Saving weights...')
    print('-'*30)
    model.save_weights(saving_weights_path+model_name+"_weights_"+str(nb_epoch)+"_epochs.h5")
    print("Saved model to disk in: "+saving_weights_path+model_name+"_weights_"+str(nb_epoch)+"_epochs.h5")

print('-'*30)
print('Model evaluation...')
print('-'*30)
score_train = model.evaluate_generator(generator=train_generator, val_samples=nb_train_samples, max_q_size=1)
print('Train Loss:', score_train[0])
print('Train Accuracy:', score_train[1])

score_test = model.evaluate_generator(generator=validation_generator, val_samples=nb_validation_samples, max_q_$
print('Test Loss:', score_test[0])
print('Test Accuracy:', score_test[1])

f_train.close()
f_test.close()
f_scores.close()
