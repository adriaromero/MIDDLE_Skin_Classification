''' Task 1: Training a network from the scratch using a Keras model'''
''' Keras model --> VGG19'''
import sys
sys.path.append('/imatge/aromero/work/image-classification/FAU_DL_imageClassification/keras-models')
from vgg19 import VGG19
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.visualize_util import plot
from keras import backend as K
K.set_image_dim_ordering('th')

# dimensions of our images.
img_width, img_height = 224, 224

train_data_dir = '/imatge/aromero/work/image-classification/isbi-dataset/train'
validation_data_dir = '/imatge/aromero/work/image-classification/isbi-dataset/test'
nb_train_samples = 896
nb_validation_samples = 312
nb_epoch = 1

print('-'*30)
print('Loading VGG19 architecture...')
print('     Downloading Imagenet weights...')
print('-'*30)
model = VGG19(weights='imagenet')

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
                    rescale=1./255,
                    shear_range=0.2,
                    rotation_range=40,
                    width_shift_range=0.2,
                    height_shift_range=0.2,
                    zoom_range=0.2,
                    horizontal_flip=True)

print('-'*30)
print('Data augmentation...')
print('-'*30)
# this is the augmentation configuration we will use for testing:
<<<<<<< Updated upstream:task1_skin.py
# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
=======
test_datagen = ImageDataGenerator(rescale=1./255)
>>>>>>> Stashed changes:keras_model_VGG19.py

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

#print('-'*30)
#print('Saving weights...')
#print('-'*30)
#model.save_weights('task1_skin_weights_' + str(nb_epoch) + 'epochs.h5')

print('-'*30)
print('Model Evaluation...')
print('-'*30)
score = model.evaluate_generator(test_generator,val_samples=20,max_q_size=10)
print('Test Loss:', score[0])
print('Test Accuracy:', score[1])
