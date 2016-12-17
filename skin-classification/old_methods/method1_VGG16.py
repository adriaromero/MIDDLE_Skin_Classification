'''method 1: training a small network from scratch'''
import os
import numpy as np
np.random.seed(2016)  # for reproducibility
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import DirectoryIterator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, BatchNormalization
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils.visualize_util import plot
from keras.utils import np_utils

# Options
SAVE_WEIGHTS = 0
PRINT_MODEL = 0

# Paths to set
model_name = "method1_VGG16"
model_path = "models_trained/" +model_name+"/"
weights_path = "models_trained/"+model_name+"/weights/"
train_data_dir = '/imatge/aromero/work/image-classification/balanced-isbi-classification-dataset/train'
validation_data_dir = '/imatge/aromero/work/image-classification/balanced-isbi-classification-dataset/val'

# Network Parameters
img_width, img_height = 224, 224	# Image dimensions
nb_train_samples = 900				# Training samples
nb_validation_samples = 150			# Testing samples
batch_size = 16
nb_epoch = 50
dropout = 0.5

# Create directories for the models
if not os.path.exists(model_path):
	os.makedirs(model_path)
	os.makedirs(weights_path)

# Initialize result files
#f_model = open(model_path+model_name+"_model.txt", 'w')
f_hist = open(model_path+model_name+"_history.txt", 'w')
f_train = open(model_path+model_name+"_train.txt", 'w')
f_test = open(model_path+model_name+"_test.txt", 'w')

print('-'*30)
print('Defining VGG16 architecture...')
print('-'*30)
model = Sequential()
model.add(Convolution2D(32, 3, 3, input_shape=(3, img_width, img_height)))
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
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.summary()

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

if(PRINT_MODEL):
    print('-'*30)
    print('Printing model...')
    print('-'*30)
    plot(model, to_file='method1_VGG16_model.png')

# this is the augmentation configuration use for training
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
# this is the augmentation configuration for testing:
test_datagen = ImageDataGenerator(rescale=1./255)

print('-'*30)
print('Creating train batches...')
print('-'*30)
train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary')

print ('Training classes: ', train_generator.class_indices)

print('-'*30)
print('Creating test batches...')
print('-'*30)
validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary')

print ('Validation classes: ', validation_generator.class_indices)

print('-'*30)
print('Training model...')
print('-'*30)
scores = model.fit_generator( train_generator,
                	samples_per_epoch=nb_train_samples,
                	nb_epoch=nb_epoch,
                	validation_data=validation_generator,
                	nb_val_samples=nb_validation_samples)

# Save performance on every epoch
f_hist.write(str(scores.history))

print('-'*30)
print('Model evaluation...')
print('-'*30)

score_train = model.evaluate_generator(generator=train_generator, val_samples=nb_train_samples, max_q_size=10)
print('Train Loss:', score_train[0])
print('Train Accuracy:', score_train[1])
f_train.write(str(score_train)+"\n")

score_test = model.evaluate_generator(generator=validation_generator, val_samples=nb_validation_samples, max_q_size=10)
print('Test Loss:', score_test[0])
print('Test Accuracy:', score_test[1])
f_test.write(str(score_test)+"\n")

if(SAVE_WEIGHTS):
	print('-'*30)
	print('Saving weights...')
	print('-'*30)
	model.save_weights(weights_path+model_name+"_weights.h5")


#f_model.close()
f_hist.close()
f_train.close()
f_test.close()

# Check generator
'''iterator = DirectoryIterator(directory=train_data_dir, image_data_generator=train_generator,
                 target_size=(img_width, img_height),
				 class_mode='binary')
(X_train, y_train) = iterator.next(train_generator)

# Predict test generator
y_pred = model.predict_generator(validation_generator, nb_validation_samples)
np.savetxt('y_pred.txt', y_pred)'''
