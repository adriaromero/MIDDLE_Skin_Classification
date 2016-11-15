import os
from keras.models import Model
from keras.layers import Input, Activation, merge, Dense, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.visualize_util import plot
from keras import backend as K
K.set_image_dim_ordering('th')

# Options
SAVE_WEIGHTS = 1
PRINT_MODEL = 1

# dimensions of our images.
img_width, img_height = 224, 224

# Paths to set
model_name = "resnet_train"
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


# Helper to build a conv -> BN -> relu block
def _conv_bn_relu(nb_filter, nb_row, nb_col, subsample=(1, 1)):
    def f(input):
        conv = Convolution2D(nb_filter=nb_filter, nb_row=nb_row, nb_col=nb_col, subsample=subsample,
                             init="he_normal", border_mode="same")(input)
        norm = BatchNormalization(mode=0, axis=1)(conv)
        return Activation("relu")(norm)

    return f


# Helper to build a BN -> relu -> conv block
# This is an improved scheme proposed in http://arxiv.org/pdf/1603.05027v2.pdf
def _bn_relu_conv(nb_filter, nb_row, nb_col, subsample=(1, 1)):
    def f(input):
        norm = BatchNormalization(mode=0, axis=1)(input)
        activation = Activation("relu")(norm)
        return Convolution2D(nb_filter=nb_filter, nb_row=nb_row, nb_col=nb_col, subsample=subsample,
                             init="he_normal", border_mode="same")(activation)

    return f


# Adds a shortcut between input and residual block and merges them with "sum"
def _shortcut(input, residual):
    # Expand channels of shortcut to match residual.
    # Stride appropriately to match residual (width, height)
    # Should be int if network architecture is correctly configured.
    stride_width = input._keras_shape[2] // residual._keras_shape[2]
    stride_height = input._keras_shape[3] // residual._keras_shape[3]
    equal_channels = residual._keras_shape[1] == input._keras_shape[1]

    shortcut = input
    # 1 X 1 conv if shape is different. Else identity.
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        shortcut = Convolution2D(nb_filter=residual._keras_shape[1], nb_row=1, nb_col=1,
                                 subsample=(stride_width, stride_height),
                                 init="he_normal", border_mode="valid")(input)

    return merge([shortcut, residual], mode="sum")


# Builds a residual block with repeating bottleneck blocks.
def _residual_block(block_function, nb_filters, repetitions, is_first_layer=False):
    def f(input):
        for i in range(repetitions):
            init_subsample = (1, 1)
            if i == 0 and not is_first_layer:
                init_subsample = (2, 2)
            input = block_function(nb_filters=nb_filters, init_subsample=init_subsample)(input)
        return input

    return f


# Basic 3 X 3 convolution blocks.
# Use for resnet with layers <= 34
# Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
def basic_block(nb_filters, init_subsample=(1, 1)):
    def f(input):
        conv1 = _bn_relu_conv(nb_filters, 3, 3, subsample=init_subsample)(input)
        residual = _bn_relu_conv(nb_filters, 3, 3)(conv1)
        return _shortcut(input, residual)

    return f


# Bottleneck architecture for > 34 layer resnet.
# Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
# Returns a final conv layer of nb_filters * 4
def bottleneck(nb_filters, init_subsample=(1, 1)):
    def f(input):
        conv_1_1 = _bn_relu_conv(nb_filters, 1, 1, subsample=init_subsample)(input)
        conv_3_3 = _bn_relu_conv(nb_filters, 3, 3)(conv_1_1)
        residual = _bn_relu_conv(nb_filters * 4, 1, 1)(conv_3_3)
        return _shortcut(input, residual)

    return f


class ResNetBuilder(object):
    @staticmethod
    def build(input_shape, num_outputs, block_fn, repetitions):
        """
        Builds a custom ResNet like architecture.
        :param input_shape: The input shape in the form (nb_channels, nb_rows, nb_cols)
        :param num_outputs: The number of outputs at final softmax layer
        :param block_fn: The block function to use. This is either :func:`basic_block` or :func:`bottleneck`.
        The original paper used basic_block for layers < 50
        :param repetitions: Number of repetitions of various block units.
        At each block unit, the number of filters are doubled and the input size is halved
        :return: The keras model.
        """
        if len(input_shape) != 3:
            raise Exception("Input shape should be a tuple (nb_channels, nb_rows, nb_cols)")

        input = Input(shape=input_shape)
        conv1 = _conv_bn_relu(nb_filter=64, nb_row=7, nb_col=7, subsample=(2, 2))(input)
        pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), border_mode="same")(conv1)

        block = pool1
        nb_filters = 64
        for i, r in enumerate(repetitions):
            block = _residual_block(block_fn, nb_filters=nb_filters, repetitions=r, is_first_layer=i == 0)(block)
            nb_filters *= 2 # nb_filters=128

        # Classifier block
        pool2 = AveragePooling2D(pool_size=(block._keras_shape[2], block._keras_shape[3]), strides=(1, 1))(block)
        flatten1 = Flatten()(pool2)
        dense = Dense(output_dim=num_outputs, init="he_normal", activation="softmax")(flatten1)

        model = Model(input=input, output=dense)
        return model

    @staticmethod
    def build_resnet_18(input_shape, num_outputs):
        return ResNetBuilder.build(input_shape, num_outputs, basic_block, [2, 2, 2, 2])

    @staticmethod
    def build_resnet_34(input_shape, num_outputs):
        return ResNetBuilder.build(input_shape, num_outputs, basic_block, [3, 4, 6, 3])

    @staticmethod
    def build_resnet_50(input_shape, num_outputs):
        return ResNetBuilder.build(input_shape, num_outputs, bottleneck, [3, 4, 6, 3])

    @staticmethod
    def build_resnet_101(input_shape, num_outputs):
        return ResNetBuilder.build(input_shape, num_outputs, bottleneck, [3, 4, 23, 3])

    @staticmethod
    def build_resnet_152(input_shape, num_outputs):
        return ResNetBuilder.build(input_shape, num_outputs, bottleneck, [3, 8, 36, 3])


def main():
    model = ResNetBuilder.build_resnet_18(input_shape=(3, img_width, img_height), num_outputs=1)
    model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=['accuracy'])
    model.summary()
    if(PRINT_MODEL):
        print('-'*30)
        print('Printing model...')
        print('-'*30)
        plot(model, to_file='resnet_model.png')

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

        if(SAVE_WEIGHTS):
            print('-'*30)
            print('Saving weights...')
            print('-'*30)
            model.save_weights(weights_path+model_name+"_weights_epoch"+str(epoch)+".h5")
            print("Saved model to disk in: "+weights_path+model_name+"_weights_epoch"+str(epoch)+".h5")

        score_train = model.evaluate_generator(generator=train_generator, val_samples=nb_train_samples, max_q_size=1)
        f_train.write(str(score_train)+"\n")

        score_test = model.evaluate_generator(generator=validation_generator, val_samples=nb_validation_samples, max_q_size=1)
        f_test.write(str(score_test)+"\n")

        f_scores.write(str(score_train[0])+","+str(score_train[1])+","+str(score_test[0])+","+str(score_test[1])+"\n")

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
