import os
from keras.models import Model
from keras.layers import Input, Activation, merge, Dense, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.data_utils import get_file
from keras.utils.visualize_util import plot
from keras import backend as K
K.set_image_dim_ordering('th')

# Options
SAVE_WEIGHTS = 1
PRINT_MODEL = 1

# Paths
TH_WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_th_dim_ordering_th_kernels.h5'

def _conv_bn_relu(nb_filter, nb_row, nb_col, subsample=(1, 1)):
    def f(input):
        conv = Convolution2D(nb_filter=nb_filter, nb_row=nb_row, nb_col=nb_col, subsample=subsample,
                             init="he_normal", border_mode="same")(input)
        norm = BatchNormalization(mode=0, axis=1)(conv)
        return Activation("relu")(norm)
    return f

def _bn_relu_conv(nb_filter, nb_row, nb_col, subsample=(1, 1)):
    def f(input):
        norm = BatchNormalization(mode=0, axis=1)(input)
        activation = Activation("relu")(norm)
        return Convolution2D(nb_filter=nb_filter, nb_row=nb_row, nb_col=nb_col, subsample=subsample,
                             init="he_normal", border_mode="same")(activation)
    return f

def _basic_block(nb_filters, init_subsample=(1, 1)):
    def f(input):
        conv1 = _bn_relu_conv(nb_filters, 3, 3, subsample=init_subsample)(input)
        residual = _bn_relu_conv(nb_filters, 3, 3)(conv1)
        return _shortcut(input, residual)
    return f

def _shortcut(input, residual):
    stride_width = input._keras_shape[2] / residual._keras_shape[2]
    stride_height = input._keras_shape[3] / residual._keras_shape[3]
    equal_channels = residual._keras_shape[1] == input._keras_shape[1]

    shortcut = input
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        shortcut = Convolution2D(nb_filter=residual._keras_shape[1], nb_row=1, nb_col=1,
                                 subsample=(stride_width, stride_height),
                                 init="he_normal", border_mode="valid")(input)

    return merge([shortcut, residual], mode="sum")

def _residual_block(block_function, nb_filters, repetations, is_first_layer=False):
    def f(input):
        for i in range(repetations):
            init_subsample = (1, 1)
            if i == 0 and not is_first_layer:
                init_subsample = (2, 2)
            input = block_function(nb_filters=nb_filters, init_subsample=init_subsample)(input)
        return input
    return f

def resnet():
    input = Input(shape=(3, 224, 224))

    conv1 = _conv_bn_relu(nb_filter=64, nb_row=7, nb_col=7, subsample=(2, 2))(input)
    pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), border_mode="same")(conv1)

    # Build residual blocks..
    block_fn = _basic_block
    block1 = _residual_block(block_fn, nb_filters=64, repetations=3, is_first_layer=True)(pool1)
    block2 = _residual_block(block_fn, nb_filters=128, repetations=4)(block1)
    block3 = _residual_block(block_fn, nb_filters=256, repetations=6)(block2)
    block4 = _residual_block(block_fn, nb_filters=512, repetations=3)(block3)

    # Classifier block
    pool2 = AveragePooling2D(pool_size=(7, 7), strides=(1, 1), border_mode="same")(block4)
    flatten1 = Flatten()(pool2)
    dense = Dense(output_dim=1, init="he_normal", activation="softmax")(flatten1)

    model = Model(input=input, output=dense)
    return model

def main():

    model = resnet()

    # Load Imagenet weights
    print('-'*30)
    print('Loading Imagenet weights...')
    print('-'*30)
    weights_path = get_file('resnet50_weights_th_dim_ordering_th_kernels.h5',
                            TH_WEIGHTS_PATH,
                            cache_subdir='models',
                            md5_hash='1c1f8f5b0c8ee28fe9d950625a230e1c')

    model.load_weights(weights_path)

    # dimensions of our images.
    img_width, img_height = 224, 224

    # Paths to set
    model_name = "resnet_china"
    model_path = "models_trained/" +model_name+"/"
    weights_path = "models_trained/"+model_name+"/weights/"
    train_data_dir = '/imatge/aromero/work/image-classification/isbi-dataset/train'
    validation_data_dir = '/imatge/aromero/work/image-classification/isbi-dataset/test'

    # Network Parameters
    nb_train_samples = 896
    nb_validation_samples = 312
    batch_size = 32
    nb_epoch = 50

    # Create directories for the models
    if not os.path.exists(model_path):
    	os.makedirs(model_path)
    	os.makedirs(weights_path)

    # Initialize result files
    f_train = open(model_path+model_name+"_scores_training.txt", 'w')
    f_test = open(model_path+model_name+"_scores_test.txt", 'w')
    f_scores = open(model_path+model_name+"_scores.txt", 'w')

    model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=['accuracy'])
    model.summary()

    if(PRINT_MODEL):
        print('-'*30)
        print('Printing model...')
        print('-'*30)
        plot(model, to_file='resnet_china_model.png')

    # this is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator(
                        rescale=1./255,
                        shear_range=0.2,
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

        print ("Number of epoch: " +str(epoch)+"/"+str(nb_epoch))

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

        f_scores.write(str(score_train[0])+","+str(score_train[1])+","+str(score_test[0])+","+str(score_test[1])+"\n")

    if(SAVE_WEIGHTS):
        print('-'*30)
        print('Saving weights...')
        print('-'*30)
        model.save_weights(weights_path+model_name+"_weights_epoch.h5")

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

if __name__ == '__main__':
    main()
