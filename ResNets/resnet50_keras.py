'''Using ResNet50 to classify images'''
import os
import h5py
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import merge, Input
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import BatchNormalization
from keras.utils.visualize_util import plot
from keras.utils.data_utils import get_file
import matplotlib.pyplot as plt
from keras import backend as K
K.set_image_dim_ordering('th')

# Options
SAVE_WEIGHTS = 0
PRINT_MODEL = 0

# Paths to set
model_name = "resnet50_keras"
model_path = "models_trained/" +model_name+"/"
weights_path = "models_trained/"+model_name+"/weights/"
train_data_dir = '/imatge/aromero/work/image-classification/isbi-classification-dataset/train'
validation_data_dir = '/imatge/aromero/work/image-classification/isbi-classification-dataset/val'

#TH_WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_th_dim_ordering_th_kernels.h5'

# dimensions of our images.
img_width, img_height = 224, 224

# Network Parameters
nb_train_samples = 900
nb_validation_samples = 378
batch_size = 16
nb_epoch = 100

# Create directories for the models
if not os.path.exists(model_path):
	os.makedirs(model_path)
	os.makedirs(weights_path)

# Initialize result files
f_train = open(model_path+model_name+"_scores_training.txt", 'w')
f_test = open(model_path+model_name+"_scores_test.txt", 'w')
f_scores = open(model_path+model_name+"_scores.txt", 'w')

# Keras ResNet50 previous functions
def identity_block(input_tensor, kernel_size, filters, stage, block):
    '''The identity_block is the block that has no conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    '''
    nb_filter1, nb_filter2, nb_filter3 = filters
    bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Convolution2D(nb_filter1, 1, 1, name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter2, kernel_size, kernel_size,
                      border_mode='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter3, 1, 1, name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = merge([x, input_tensor], mode='sum')
    x = Activation('relu')(x)
    return x

def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    '''conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
    And the shortcut should have subsample=(2,2) as well
    '''
    nb_filter1, nb_filter2, nb_filter3 = filters
    bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Convolution2D(nb_filter1, 1, 1, subsample=strides,
                      name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter2, kernel_size, kernel_size, border_mode='same',
                      name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter3, 1, 1, name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Convolution2D(nb_filter3, 1, 1, subsample=strides,
                             name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = merge([x, shortcut], mode='sum')
    x = Activation('relu')(x)
    return x

# Start defining ResNet50 and loading ImageNet weights
img_input = Input(shape=(3,img_width, img_height))
bn_axis = 1

x = ZeroPadding2D((3, 3))(img_input)
x = Convolution2D(64, 7, 7, subsample=(2, 2), name='conv1')(x)
x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
x = Activation('relu')(x)
x = MaxPooling2D((3, 3), strides=(2, 2))(x)

x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

x = AveragePooling2D((7, 7), name='avg_pool')(x)

x = Flatten()(x)
x = Dense(1, activation='softmax', name='fc1')(x)

model = Model(img_input, x)

# load weights
#print('-'*30)
#print('Loading Imagenet weights...')
#print('-'*30)
#weights_path = get_file('resnet50_weights_th_dim_ordering_th_kernels.h5',
#                        TH_WEIGHTS_PATH,
#                        cache_subdir='models',
#                        md5_hash='1c1f8f5b0c8ee28fe9d950625a230e1c')
#model.load_weights(weights_path)

print('-'*30)
print('Model summary...')
print('-'*30)
model.summary()

if(PRINT_MODEL):
    print('-'*30)
    print('Printing model...')
    print('-'*30)
    plot(model, to_file='resnet50_keras_model.png')

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

    scores = model.fit_generator(train_generator,
    				samples_per_epoch=nb_train_samples,
                    nb_epoch=1,
                    validation_data=validation_generator,
                    nb_val_samples=nb_validation_samples)

    score_train = model.evaluate_generator(generator=train_generator, val_samples=nb_train_samples, max_q_size=10)
    f_train.write(str(score_train)+"\n")

    score_test = model.evaluate_generator(generator=validation_generator, val_samples=nb_validation_samples, max_q_size=10)
    f_test.write(str(score_test)+"\n")

    f_scores.write(str(score_train[0])+","+str(score_train[1])+","+str(score_test[0])+","+str(score_test[1])+"\n")

if(SAVE_WEIGHTS):
	print('-'*30)
	print('Saving weights...')
	print('-'*30)
	model.save_weights(weights_path+model_name+"_weights_"+str(nb_epoch)+"epochs.h5")

print('-'*30)
print('Model evaluation...')
print('-'*30)
score_train = model.evaluate_generator(generator=train_generator, val_samples=nb_train_samples, max_q_size=10)
print('Train Loss:', score_train[0])
print('Train Accuracy:', score_train[1])

score_test = model.evaluate_generator(generator=validation_generator, val_samples=nb_validation_samples, max_q_size=10)
print('Test Loss:', score_test[0])
print('Test Accuracy:', score_test[1])

f_train.close()
f_test.close()
f_scores.close()
