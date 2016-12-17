'''method 3: fine-tuning the model'''
import os
import h5py
import numpy as np
np.random.seed(2016)  # for reproducibility
import matplotlib.pyplot as plt
import matplotlib
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, BatchNormalization
from keras.layers import Activation, Dropout, Flatten, Dense, ZeroPadding2D
from keras.utils.visualize_util import plot
from keras.utils import np_utils
from keras.preprocessing import image
from keras.optimizers import SGD

# Options
SAVE_WEIGHTS = 1
PRINT_MODEL = 0

# Paths to set
model_name = "method3_VGG16"
model_path = "models_trained/" +model_name+"/"
saving_weights_path = "models_trained/"+model_name+"/weights/"
weights_path = '/imatge/aromero/work/image-classification/weights/vgg16_weights.h5'
top_model_weights_path = '/imatge/aromero/work/image-classification/MIDDLE_Skin_Classification/skin-classification/classical_loading/models_trained/method1_VGG16/weights/method1_VGG16_weights.h5'
train_data_dir = '/imatge/aromero/work/image-classification/isbi-classification-dataset/train'
validation_data_dir = '/imatge/aromero/work/image-classification/isbi-classification-dataset/val'

# Network Parameters
img_width, img_height = 224, 224

nb_train_samples = 900
nb_validation_samples = 378
nb_epoch = 30
batch_size = 32
dropout = 0.5
freeze = 25

# Create directories for the models
if not os.path.exists(model_path):
	os.makedirs(model_path)
	os.makedirs(weights_path)

# Initialize result files
f_model = open(model_path+model_name+"_model.txt", 'w')
f_hist = open(model_path+model_name+"_history.txt", 'w')

# Load Data
def load_data(data_type):
    '''Load and resize data'''
    print 'Loading data: ', data_type
    if data_type == 'train':
        data_dir = train_data_dir
    else:
        data_dir = validation_data_dir

    # assume malignant = 0, benign = 1
    malignant_path = os.path.join(data_dir, 'malignant')
    malignant_list = os.listdir(malignant_path)  # get a list of all malignant image files in directory
    malignant_num = len(malignant_list)
    benign_path = os.path.join(data_dir, 'benign')
    benign_list = os.listdir(benign_path)
    benign_num = len(benign_list)

    _X = np.empty((benign_num + malignant_num, 3, img_width, img_height), dtype='uint8')
    _y = np.zeros((benign_num + malignant_num, ), dtype='uint8')

    # store the malignant
    for i, malignant_file in enumerate(malignant_list):
        img = image.load_img(os.path.join(malignant_path, malignant_file), grayscale=False, target_size=(img_width, img_height))
        _X[i] = image.img_to_array(img)
    # add the benign and set flag to 1 (this should be equal to "1D binary labels" as in the example flow_from_directory)
    for i, benign_file in enumerate(benign_list):
        img = image.load_img(os.path.join(benign_path, benign_file), grayscale=False, target_size=(img_width, img_height))
        _X[i + malignant_num] = image.img_to_array(img)
        _y[i + malignant_num] = 1
    return _X, _y

X_train, y_train = load_data('train')
X_test, y_test = load_data('valid')

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255


print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

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
top_model = Sequential()
top_model.add(Flatten(input_shape=model.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
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
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy',
			        optimizer=sgd,
			        metrics=['binary_accuracy', 'precision', 'recall'])

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
datagen.fit(X_train)

# fine-tune the model
print('-'*30)
print('Fine-tunning the model...')
print('-'*30)
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
    model.save_weights(saving_weights_path+model_name+"_weights.h5")
    print("Saved model to disk in: "+saving_weights_path+model_name+"_weights.h5")

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
