'''method 1: training a small network from scratch'''
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, BatchNormalization
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils.visualize_util import plot
from keras.utils import np_utils
from keras.optimizers import SGD
from keras import optimizers
from keras.preprocessing import image
from keras import backend as K
K.set_image_dim_ordering('th')
import cv2

# Options
SAVE_WEIGHTS = 1
PRINT_MODEL = 0

# dimensions of our images.
img_width, img_height = 224, 224
#img_width, img_height = 500, 376	# Keeping same rotation_range
#img_width, img_height = 340, 255

# Paths to set
model_name = "method1_VGG16"
model_path = "models_trained/" +model_name+"/"
weights_path = "models_trained/"+model_name+"/weights/"
#train_data_dir = '/imatge/aromero/work/image-classification/isbi-classification-dataset/train'
#validation_data_dir = '/imatge/aromero/work/image-classification/isbi-classification-dataset/val'
train_data_dir = '/imatge/aromero/work/image-classification/dogs_cats_dataset/train'
validation_data_dir = '/imatge/aromero/work/image-classification/dogs_cats_dataset/val'

# Network Parameters
#nb_train_samples = 900
#nb_train_samples = 900
nb_train_samples = 2000
nb_train_samples = 800
batch_size = 32
nb_epoch = 5

# Create directories for the models
if not os.path.exists(model_path):
	os.makedirs(model_path)
	os.makedirs(weights_path)

# Initialize result files
f_hist = open(model_path+model_name+"_history.txt", 'w')
f_model = open(model_path+model_name+"_model.txt", 'w')

# Load Data
def load_data(data_type):
    '''Load and resize data'''
    print 'Loading data: ', data_type
    if data_type == 'train':
        data_dir = train_data_dir
    else:
        data_dir = validation_data_dir

    # assume malignant = 0, benign = 1
    #malignant_path = os.path.join(data_dir, 'malignant')
    malignant_path = os.path.join(data_dir, 'cats')
    malignant_list = os.listdir(malignant_path)  # get a list of all malignant image files in directory
    malignant_num = len(malignant_list)
    #benign_path = os.path.join(data_dir, 'benign')
    benign_path = os.path.join(data_dir, 'dogs')
    benign_list = os.listdir(benign_path)
    benign_num = len(benign_list)

    #_X = np.empty((benign_num + malignant_num, 3, img_width, img_height), dtype='float32')
    _X = np.empty((benign_num + malignant_num, 3, img_width, img_height), dtype='uint8')
    _y = np.zeros((benign_num + malignant_num, ), dtype='uint8')

    # store the malignant
    i = 0
    for malignant_file in malignant_list:
	#img = image.load_img(os.path.join(malignant_path, malignant_file), grayscale=False, target_size=(img_width,img_height))
	img = cv2.imread(os.path.join(malignant_path, malignant_file))
	img = cv2.resize(img, (img_width,img_height))
    	_X[i] = image.img_to_array(img)
	i = i + 1

    i = 0
    # add the benign and set flag to 1 (this should be equal to "1D binary labels" as in the example flow_from_directory)
    for benign_file in benign_list:
        #img = image.load_img(os.path.join(benign_path, benign_file), grayscale=False, target_size=(img_width,img_height))
	img = cv2.imread(os.path.join(benign_path, benign_file))
	img = cv2.resize(img, (img_width,img_height))
        _X[i + malignant_num] = image.img_to_array(img)
	_y[i + malignant_num] = 1
	i = i + 1

    return _X, _y

X_train, y_train = load_data('train')
X_test, y_test = load_data('valid')

X_train = X_train.astype('float32')
X_train /= 255.0
X_test = X_test.astype('float32')
X_test /= 255.0

# Checking everything is alright
#np.savetxt('X_train.txt', X_train[1,1,:,:], delimiter=',')
#np.savetxt('y_train.txt', y_train[:], delimiter=',')
#np.savetxt('X_test.txt', X_test[1,1,:,:], delimiter=',')
#np.savetxt('y_test.txt', y_test[:], delimiter=',')
#matplotlib.image.imsave('image_1_method1.png', X_train[0,0,:,:])

print'X_train shape:', X_train.shape
print'X_test shape:', X_test.shape
print'y_train shape:', y_train.shape
print'y_test shape:', y_test.shape
print X_train.shape[0], 'train samples'
print X_test.shape[0], 'test samples'

print('-'*30)
print('Defining VGG16 architecture...')
print('-'*30)
model = Sequential()
'''model.add(Convolution2D(32, 3, 3, input_shape=(3, img_width, img_height)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(dropout))
model.add(Dense(1))
model.add(Activation('sigmoid'))'''

model.add(Convolution2D(32, 3, 3, border_mode='same',
                        input_shape=(3, img_width, img_height)))
model.add(Activation('relu'))
model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('softmax'))

model.summary()

if(PRINT_MODEL):
    print('-'*30)
    print('Printing model...')
    print('-'*30)
    plot(model, to_file='method1_VGG16_model.png')

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy',
              optimizer=sgd, 	# optimizers.SGD(lr=3e-4, momentum=0.9)
              metrics=['accuracy', 'precision', 'recall'])

print('Using real-time data augmentation')

# this will do preprocessing and realtime data augmentation
datagen = ImageDataGenerator(
	featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=40,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False)  # randomly flip images

# compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)
#datagen.fit(X_train)

scores = model.fit_generator(datagen.flow(X_train, y_train,
					batch_size=batch_size),
                    samples_per_epoch=X_train.shape[0],
                    nb_epoch=nb_epoch,
                    validation_data=(X_test, y_test))

# Save performance on every epoch
f_hist.write(str(scores.history))

if(SAVE_WEIGHTS):
	print('-'*30)
	print('Saving weights...')
	print('-'*30)
	model.save_weights(weights_path+model_name+"_weights.h5")

print('-'*30)
print('Model evaluation...')
print('-'*30)
score_train = model.evaluate(X_train, y_train, verbose=0)
print('Train Loss:', score_train[0])
print('Train Accuracy:', score_train[1])

score_test = model.evaluate(X_test, y_test, verbose=0)
print('Test Loss:', score_test[0])
print('Test Accuracy:', score_test[1])

f_model.write(str(score_train[0])+","+str(score_train[1])+","+str(score_test[0])+","+str(score_test[1])+"\n")

f_hist.close()
f_model.close()

#y_pred_labels = model.predict_classes(X_test)
#np.savetxt('y_pred.txt', y_pred_labels, delimiter=',')

#y_pred_values = model.predict(X_test)
#np.savetxt('y_pred.txt', y_pred_values, delimiter=',')
