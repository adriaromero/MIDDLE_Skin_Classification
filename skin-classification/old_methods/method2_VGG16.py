'''Method 2: using a pre-trained model'''
import os
import h5py
import numpy as np
np.random.seed(2016)  # for reproducibility
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils.visualize_util import plot

# Options
SAVE_WEIGHTS = 0
PRINT_MODEL = 0

# Paths to set
model_name = "method2_VGG16"
model_path = "models_trained/" +model_name+"/"
weights_path = "models_trained/"+model_name+"/weights/"
train_data_dir = '/imatge/aromero/work/image-classification/balanced-isbi-classification-dataset/train'
validation_data_dir = '/imatge/aromero/work/image-classification/balanced-isbi-classification-dataset/val'
saved_weights_path = '/imatge/aromero/work/image-classification/weights/vgg16_weights.h5'
top_model_weights_path = '/imatge/aromero/work/image-classification/MIDDLE_Skin_Classification/skin-classification/old_methods/models_trained/method1_VGG16/weights/method1_VGG16_weights.h5'

# dimensions of our images.
img_width, img_height = 224, 224

### other hyperparameters
nb_train_samples = 346				# Training samples
nb_train_samples_benign = 173		# Testing samples
nb_train_samples_malignant = 173	# Malignant Training samples
nb_validation_samples = 150			# Malignant Training samples
nb_validation_samples_benign = 75	# Benign Training samples
nb_validation_samples_maligant = 75	# Malignant Testing samples
nb_epoch = 50
img_width, img_height = 224, 224
batch_size = 32

# Create directories for the models
if not os.path.exists(model_path):
	os.makedirs(model_path)
	os.makedirs(weights_path)

# Initialize result files
f_model = open(model_path+model_name+"_model.txt", 'w')
f_hist = open(model_path+model_name+"_history.txt", 'w')

def save_bottlebeck_features():
    datagen = ImageDataGenerator(rescale=1./255,
								shear_range=0,
								rotation_range=40, # randomly rotate images in the range (degrees, 0 to 180)
								width_shift_range=0.2, # randomly shift images horizontally (fraction of total width)
								height_shift_range=0.2, # randomly shift images vertically (fraction of total height)
								zoom_range=0.2,
								horizontal_flip=True, # randomly flip images
								vertical_flip=True)
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

    # load the weights of the VGG16 networks (trained on ImageNet)
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
    train_labels = np.array([0] * nb_train_samples_benign + [1] * nb_train_samples_malignant)

    validation_data = np.load(open('bottleneck_features_validation.npy'))
    validation_labels = np.array([0] * nb_validation_samples_benign + [1] * nb_validation_samples_maligant)

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

    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy', 'precision', 'recall'])

    print('-'*30)
    print('Training model...')
    print('-'*30)
    score = model.fit(train_data, train_labels,
                    nb_epoch=nb_epoch, batch_size=batch_size,
                    validation_data=(validation_data, validation_labels))

    # Save performance on every epoch
    f_hist.write(str(score.history))

    print('-'*30)
    print('Saving weights...')
    print('-'*30)
    model.save_weights(top_model_weights_path)

    print('-'*30)
    print('Model Evaluation...')
    print('-'*30)
    score_train = model.evaluate(train_data, train_labels)
    print('Train Loss:', score_train[0])
    print('Train Accuracy:', score_train[1])

    score_test = model.evaluate(validation_data, validation_labels)
    print('Test Loss:', score_test[0])
    print('Test Accuracy:', score_test[1])

    f_model.write(str(score_train[0])+","+str(score_train[1])+","+str(score_test[0])+","+str(score_test[1])+"\n")

    f_model.close()
    f_hist.close()

save_bottlebeck_features()
train_top_model()
