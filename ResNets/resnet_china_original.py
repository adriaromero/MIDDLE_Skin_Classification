from keras.models import Model
from keras.layers import Input, Activation, merge, Dense, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.utils.visualize_util import plot

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
    dense = Dense(output_dim=1000, init="he_normal", activation="softmax")(flatten1)

    model = Model(input=input, output=dense)
    return model
