from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from tqdm import tqdm
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import load_data as ld
from skimage.color import rgb2gray
import argparse
import numpy as np
import keras
from tensorflow.python.platform import flags
from cleverhans.utils import save_leg_sample, save_leg_sample_rgb

FLAGS = flags.FLAGS


def set_gtsrb_flags():
    try:
        flags.DEFINE_integer('BATCH_SIZE', 32, 'Size of training batches')
    except argparse.ArgumentError:
        pass

    flags.DEFINE_integer('NUM_CLASSES', 43, 'Number of classification classes')
    flags.DEFINE_integer('IMAGE_ROWS', 32, 'Input row dimension')
    flags.DEFINE_integer('IMAGE_COLS', 32, 'Input column dimension')
    flags.DEFINE_integer('NUM_CHANNELS', 1, 'Input depth dimension')


def preprocess(image):
    image = image.astype(float)
    return (image - 255.0 / 2) / 255.0


def preprocess_batch(images):
    imgs = np.copy(images)
    for i in tqdm(range(images.shape[0])):
        imgs[i] = preprocess(images[i])
    return imgs


def load_data(train_data_dir, test_data_dir):
    """
        Load GTSRB dataset.
        :param train_data_dir: path to folder where training data should be stored.
        :param test_data_dir: path to folder where testing data should be stored.
        :return: tuple of four arrays containing training data, training labels,
                     testing data and testing labels.
    """
    X_train, y_train, X_test, y_test = ld.load_data(train_data_dir, test_data_dir)

    # X_train_norm = (X_train.astype(float))/255 - 0.5
    # X_test_norm = (X_test.astype(float))/255 - 0.5
    X_train_norm = preprocess_batch(X_train.astype(float))
    X_test_norm = preprocess_batch(X_test.astype(float))

    # Transfer RGB to gray
    X_train_transfered, X_test_transfered = rgb_convert_grayscale(X_train_norm, X_test_norm)

    X_train_transfered, X_val_transfered, y_train, y_val = train_test_split(X_train_transfered, y_train, test_size=0.2,
                                                                            random_state=42)

    print(X_train_transfered.shape, y_train.shape, X_val_transfered.shape, y_val.shape, X_test_transfered.shape, y_test.shape)

    # save_leg_sample('GTSRB/test/', X_test_norm)
    # save_leg_sample_rgb('GTSRB/test/', X_test)

    # Shuffle training data
    X_train, y_train = shuffle(X_train_transfered, y_train)
    X_val, y_val = shuffle(X_val_transfered, y_val)
    X_test, y_test = shuffle(X_test_transfered, y_test)
    return X_train, y_train, X_val, y_val, X_test, y_test


def rgb_convert_grayscale(X_train, X_test):
    """
        Convert to grayscale from RGB
        :param X_train: the training data for the oracle.
        :param Y_train: the training labels for the oracle.
        :param Label: Given label.
        :return:
    """
    train_imgs = rgb2gray(X_train)
    test_imgs = rgb2gray(X_test)
    # Reshape images to [num_samples, rows*columns]
    train_imgs = train_imgs.reshape(train_imgs.shape[0], train_imgs.shape[1], train_imgs.shape[2], 1)
    test_imgs = test_imgs.reshape(test_imgs.shape[0], test_imgs.shape[1], test_imgs.shape[2], 1)

    return train_imgs, test_imgs


# blackbox model
def modelA():
    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=(32, 32, 1),
                     activation='relu'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), padding='same',
                     activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(128, (3, 3), padding='same',
                     activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(43, activation='softmax'))
    return model


# Sub1
def modelB():
    model = Sequential()
    model.add(Convolution2D(64, 5, 5,
                            border_mode='valid',
                            input_shape=(FLAGS.IMAGE_ROWS,
                                         FLAGS.IMAGE_COLS,
                                         FLAGS.NUM_CHANNELS)))
    model.add(Activation('relu'))

    model.add(Convolution2D(64, 5, 5))
    model.add(Activation('relu'))

    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))

    model.add(Dropout(0.5))
    model.add(Dense(FLAGS.NUM_CLASSES))
    return model


# Sub2
def modelC():
    model = Sequential()
    model.add(Dropout(0.2, input_shape=(FLAGS.IMAGE_ROWS,
                                        FLAGS.IMAGE_COLS,
                                        FLAGS.NUM_CHANNELS)))
    model.add(Convolution2D(64, 8, 8,
                            subsample=(2, 2),
                            border_mode='same'))
    model.add(Activation('relu'))

    model.add(Convolution2D(128, 6, 6,
                            subsample=(2, 2),
                            border_mode='valid'))
    model.add(Activation('relu'))

    model.add(Convolution2D(128, 5, 5,
                            subsample=(1, 1)))
    model.add(Activation('relu'))

    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(FLAGS.NUM_CLASSES))
    return model


# Sub3
def modelD():
    model = Sequential()
    model.add(Convolution2D(128, 3, 3,
                            border_mode='valid',
                            input_shape=(FLAGS.IMAGE_ROWS,
                                         FLAGS.IMAGE_COLS,
                                         FLAGS.NUM_CHANNELS)))
    model.add(Activation('relu'))

    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))

    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))

    model.add(Dropout(0.5))
    model.add(Dense(FLAGS.NUM_CLASSES))
    return model


# Sub4
def modelE():
    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=(32, 32, 1),
                     activation='relu'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), padding='same',
                     activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(200, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(43, activation='softmax'))
    return model


# Sub5
def modelF():
    model = Sequential()

    model.add(Flatten(input_shape=(32, 32, 1)))

    model.add(Dense(300, init='he_normal', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(300, init='he_normal', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(300, init='he_normal', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(300, init='he_normal', activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(43))
    return model


# sub6
def modelG():
    model = Sequential()

    model.add(Conv2D(64, (3, 3), padding='same',
                     input_shape=(28, 28, 1),
                     activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), padding='same',
                     activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    return model


# Sub7
def modelH(img_rows=32, img_cols=32, nb_classes=43):
    """
    Defines the model architecture to be used by the substitute
    :param img_rows: number of rows in input
    :param img_cols: number of columns in input
    :param nb_classes: number of classes in output
    :return: keras model
    """
    model = Sequential()

    # Find out the input shape ordering
    if keras.backend.image_dim_ordering() == 'th':
        input_shape = (1, img_rows, img_cols)
    else:
        input_shape = (img_rows, img_cols, 1)

    # Define a fully connected model (it's different than the black-box)
    layers = [Flatten(input_shape=input_shape),
              Dense(200),
              Activation('relu'),
              Dropout(0.5),
              Dense(200),
              Activation('relu'),
              Dropout(0.5),
              Dense(nb_classes),
              Activation('softmax')]

    for layer in layers:
        model.add(layer)

    return model


def model_gtsrb(type=3):
    """
    Defines MNIST model using Keras sequential model
    """

    models = [modelA, modelB, modelC, modelD, modelE, modelF, modelG, modelH]

    return models[type]()


def data_gen_gtsrb(X_train):
    datagen = ImageDataGenerator()

    datagen.fit(X_train)
    return datagen


def load_model(model_path, type=0):
    try:
        with open(model_path + '.json', 'r') as f:
            json_string = f.read()
            model = model_from_json(json_string)
    except IOError:
        model = model_gtsrb(type=type)

    model.load_weights(model_path)
    return model
