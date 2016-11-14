'''Task 1: training a small network from scratch'''

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
K.set_image_dim_ordering('th')

# dimensions of our images.
img_width, img_height = 150, 150

train_data_dir = '/imatge/aromero/work/image-classification/dogs-vs-cats-dataset-reduced/train'
validation_data_dir = '/imatge/aromero/work/image-classification/dogs-vs-cats-dataset-reduced/test'
nb_train_samples = 2000
nb_validation_samples = 800
nb_epoch = 10

print('-'*30)
print('Defining network architecture...')
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

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

print('-'*30)
print('Data augmentation...')
print('-'*30)
# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)

print('-'*30)
print('Creating train batches...')
print('-'*30)
train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=32,
        class_mode='binary')

print('-'*30)
print('Creating test batches...')
print('-'*30)
validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=32,
        class_mode='binary')

print('-'*30)
print('Fitting model...')
print('-'*30)
model.fit_generator(
        train_generator,
        samples_per_epoch=nb_train_samples,
        nb_epoch=nb_epoch,
        validation_data=validation_generator,
        nb_val_samples=nb_validation_samples)

print('-'*30)
print('Saving weights...')
print('-'*30)
model.save_weights('task1_dogs_vs_cats_weights_' + str(nb_epoch) + 'epochs.h5')
