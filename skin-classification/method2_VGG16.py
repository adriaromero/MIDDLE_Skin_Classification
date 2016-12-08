'''Method 2: using a pre-trained model'''
import os
import h5py
import numpy as np
import random
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils.visualize_util import plot
import matplotlib.pyplot as plt
from keras import backend as K
K.set_image_dim_ordering('th')
random.seed(333)

# Options
SAVE_WEIGHTS = 1
PRINT_MODEL = 0

# Paths to set
model_name = "method2_VGG16"
model_path = "models_trained/" +model_name+"/"
weights_path = "models_trained/"+model_name+"/weights/"
train_data_dir = '/imatge/aromero/work/image-classification/isbi-dataset/train'
validation_data_dir = '/imatge/aromero/work/image-classification/isbi-dataset/test'
saved_weights_path = '/imatge/aromero/work/image-classification/FAU_DL_imageClassification/weights/vgg16_weights.h5'
top_model_weights_path = '/imatge/aromero/work/image-classification/FAU_DL_imageClassification/weights/method1_skin_weights_20epochs.h5'

# dimensions of our images.
img_width, img_height = 512, 384

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

def save_bottlebeck_features():
    datagen = ImageDataGenerator(rescale=1./255,
								shear_range=0,
								rotation_range=40, # randomly rotate images in the range (degrees, 0 to 180)
								width_shift_range=0.2, # randomly shift images horizontally (fraction of total width)
								height_shift_range=0.2, # randomly shift images vertically (fraction of total height)
								zoom_range=0.2,
								horizontal_flip=True, # randomly flip images
								vertical_flip=False)
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
    assert os.path.exists(saved_weights_path), 'Model weights not found (see "saved_weights_path" variable in script).'
    f = h5py.File(saved_weights_path)
    for k in range(f.attrs['nb_layers']):
        if k >= len(model.layers):
            # we don't look at the last (fully-connected) layers in the savefile
            break
        g = f['layer_{}'.format(k)]
        weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
        model.layers[k].set_weights(weights)
    f.close()
    print('Model loaded.')

    generator = datagen.flow_from_directory(
            train_data_dir,
            target_size=(img_width, img_height),
            batch_size=batch_size,
            class_mode=None,
            shuffle=False)
    bottleneck_features_train = model.predict_generator(generator, nb_train_samples)
    np.save(open('bottleneck_features_train.npy', 'w'), bottleneck_features_train)

    generator = datagen.flow_from_directory(
            validation_data_dir,
            target_size=(img_width, img_height),
            batch_size=batch_size,
            class_mode=None,
            shuffle=False)
    bottleneck_features_validation = model.predict_generator(generator, nb_validation_samples)
    np.save(open('bottleneck_features_validation.npy', 'w'), bottleneck_features_validation)


def train_top_model():
    print('-'*30)
    print('Loading bottleneck_features_train values...')
    print('-'*30)
    train_data = np.load(open('bottleneck_features_train.npy'))
    train_labels = np.array([0] * (nb_train_samples / 2) + [1] * (nb_train_samples / 2))

    validation_data = np.load(open('bottleneck_features_validation.npy'))
    validation_labels = np.array([0] * (nb_validation_samples / 2) + [1] * (nb_validation_samples / 2))

    print('-'*30)
    print('Defining the top model architecture...')
    print('-'*30)
    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.summary()

    if(PRINT_MODEL):
        print('-'*30)
        print('Printing model...')
        print('-'*30)
        plot(model, to_file='method2_VGG16_model.png')

    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

    print('-'*30)
    print('Training model...')
    print('-'*30)
    for epoch in range(1,nb_epoch+1):

        score = model.fit(train_data, train_labels,
                    nb_epoch=1, batch_size=batch_size,
                    validation_data=(validation_data, validation_labels))
        score_train = model.evaluate(train_data, train_labels,  verbose=0)
        f_train.write(str(score_train)+"\n")

        score_test = model.evaluate(validation_data, validation_labels,  verbose=0)
        f_test.write(str(score_test)+"\n")

        f_scores.write(str(score_train[0])+","+str(score_train[1])+","+str(score_test[0])+","+str(score_test[1])+"\n")

    print('-'*30)
    print('Saving weights...')
    print('-'*30)
    model.save_weights(top_model_weights_path)

    print('-'*30)
    print('Model Evaluation...')
    print('-'*30)
    score = model.evaluate(train_data, train_labels)
    print('Train Loss:', score[0])
    print('Train Accuracy:', score[1])

    score = model.evaluate(validation_data, validation_labels)
    print('Test Loss:', score[0])
    print('Test Accuracy:', score[1])

    f_train.close()
    f_test.close()
    f_scores.close()

	#generate_results(validation_labels, y_score)

save_bottlebeck_features()
train_top_model()
