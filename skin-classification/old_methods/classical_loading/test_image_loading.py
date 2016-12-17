'''method 1: training a small network from scratch'''
import os
import numpy as np
np.random.seed(2016)  # for reproducibility
import matplotlib.pyplot as plt
import matplotlib
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, BatchNormalization
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils.visualize_util import plot
from keras.utils import np_utils
from keras import optimizers
from keras.preprocessing import image
from keras import backend as K
K.set_image_dim_ordering('th')

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
train_data_dir = '/imatge/aromero/work/image-classification/isbi-classification-dataset/train'
validation_data_dir = '/imatge/aromero/work/image-classification/isbi-classification-dataset/val'

# Network Parameters
nb_train_samples = 900
nb_validation_samples = 378
batch_size = 16
nb_epoch = 10
dropout = 0.5

# Load Data
def load_data(data_type):
    '''Load and resize data'''
    print 'Loading data: ', data_type
    if data_type == 'train':
        data_dir = train_data_dir
        print 'Loading train data... '
    else:
        data_dir = validation_data_dir
        print 'Loading test data... '

    # assume malignant = 0, benign = 1
    malignant_path = os.path.join(data_dir, 'malignant')
    malignant_list = os.listdir(malignant_path)  # get a list of all malignant image files in directory
    malignant_num = len(malignant_list)
    benign_path = os.path.join(data_dir, 'benign')
    benign_list = os.listdir(benign_path)
    benign_num = len(benign_list)

    _X = np.empty((benign_num + malignant_num, 3, img_width, img_height), dtype='float32')
    _y = np.zeros((benign_num + malignant_num, ), dtype='uint8')

    # store the malignant
    for i, malignant_file in enumerate(malignant_list):
        img = image.load_img(os.path.join(malignant_path, malignant_file), grayscale=False, target_size=(img_width,img_height))
        print(str(os.path.join(malignant_path, malignant_file)))
    	_X[i] = image.img_to_array(img)
        print(i)
    # add the benign and set flag to 1 (this should be equal to "1D binary labels" as in the example flow_from_directory)
    for i, benign_file in enumerate(benign_list):
        img = image.load_img(os.path.join(benign_path, benign_file), grayscale=False, target_size=(img_width,img_height))
        print(str(os.path.join(benign_path, benign_file)))
        _X[i + malignant_num] = image.img_to_array(img)
        print(i + malignant_num)
        _y[i + malignant_num] = 1
    return _X, _y

X_train, y_train = load_data('train')
X_test, y_test = load_data('valid')

print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

matplotlib.image.imsave('image_1.png', X_train[0,0,:,:])
