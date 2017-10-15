import os
import csv
import pickle
import warnings
import skimage.data
import skimage.transform
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from urllib import urlretrieve
from os.path import isfile
from tqdm import tqdm


# class DLProgress(tqdm):
#     last_block = 0
#
#     def hook(self, block_num=1, block_size=1, total_size=None):
#         self.total = total_size
#         self.update((block_num - self.last_block) * block_size)
#         self.last_block = block_num
#
# if not isfile('train.p'):
#     with DLProgress(unit='B', unit_scale=True, miniters=1, desc='Train Dataset') as pbar:
#         urlretrieve(
#             'https://s3.amazonaws.com/udacity-sdc/datasets/german_traffic_sign_benchmark/train.p',
#             'train.p',
#             pbar.hook)
#
# if not isfile('test.p'):
#     with DLProgress(unit='B', unit_scale=True, miniters=1, desc='Test Dataset') as pbar:
#         urlretrieve(
#             'https://s3.amazonaws.com/udacity-sdc/datasets/german_traffic_sign_benchmark/test.p',
#             'test.p',
#             pbar.hook)
#
# print('Training and Test data downloaded.')

def load_data(data_train_dir, data_test_dir):
    """
    Loads datasets and returns two lists:
        images: a list of Numpy arrays, each representing an image.
        labels: a list of numbers that represent the images labels.
    """
    warnings.filterwarnings('ignore')
    # Load Data
    training_file = data_train_dir
    testing_file = data_test_dir
    with open(training_file, mode='rb') as f:
        train = pickle.load(f)
    with open(testing_file, mode='rb') as f:
        test = pickle.load(f)

    X_train, y_train = train['features'], train['labels']
    X_test, y_test = test['features'], test['labels']
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    return X_train, y_train, X_test, y_test


def load_train_data(data_dir):
    """
    Loads a training data set and returns two lists:
        images: a list of Numpy arrays, each representing an image.
        labels: a list of numbers that represent the images labels.
    """
    # Get all subdirectories of data_dir. Each represents a label.
    directories = [d for d in os.listdir(data_dir)
                   if os.path.isdir(os.path.join(data_dir, d))]
    # Loop through the label directories and collect the data in.
    # two lists, labels and images.
    labels = []
    images = []
    for d in directories:
        label_dir = os.path.join(data_dir, d)
        file_names = [os.path.join(label_dir, f)
                      for f in os.listdir(label_dir) if f.endswith(".ppm")]
        # For each label, load it's images and add them to the images list.
        # And add the label number (i.e. directory name) to the labels list.
        for f in file_names:
            images.append(skimage.data.imread(f))
            labels.append(int(d))
    # Resize images to 32 x 32
    images32 = [skimage.transform.resize(image, (32, 32))
                for image in images]
    return images32, labels


def load_test_data(data_dir):
    """
    Loads a training data set and returns two lists:
        images: a list of Numpy arrays, each representing an image.
        labels: a list of numbers that represent the images labels.
    """
    labels = []
    images = []
    prefix = data_dir + '/'
    gtFile = open(prefix + 'GT-final_test' + '.csv')
    gtReader = csv.reader(gtFile, delimiter=';')  # csv parser for annotations file
    next(gtReader)
    # loop over all images in current annotations file
    for row in gtReader:
        images.append(skimage.data.imread(prefix + row[0]))
        labels.append(row[7])
    gtFile.close()
    # Resize images to 32 x 32
    images32 = [skimage.transform.resize(image, (32, 32))
                for image in images]
    return images32, labels


def display_images_labels(images, labels):
    """Display the first image of each label"""
    unique_labels = set(labels)
    print(unique_labels)
    plt.figure(figsize=(25, 25))
    i = 1
    for label in unique_labels:
        # Pick the first image for each label.
        print(label)
        image = images[labels.index(label)]
        plt.subplot(8, 8, i)  # A grid of 8 rows x 8 columns.
        plt.axis("off")
        plt.title("Label {0} ({1})".format(label, labels.count(label)))
        i += 1
        _ = plt.imshow(image)
    plt.show()


def display_label_images(images, labels, label):
    """Display images of a specific label."""
    print(labels)
    limit = 24  # show a max of 24 images.
    plt.figure(figsize=(15, 5))
    i = 1
    start = labels.index(label)
    end = start + labels.count(label)
    for image in images[start:end][:limit]:
        plt.subplot(3, 8, i)  # 3 rows, 8 per row.
        plt.axis('off')
        i += 1
        plt.imshow(image)
    plt.show()


def display_one_image(X_test, y_test):
    """Test if test data load successfully"""
    print(y_test[:10])
    print(y_test[1])
    plt.figure(figsize=(2, 2))
    plt.imshow(X_test[1])
    plt.show()


if __name__ == '__main__':
    ROOT_PATH = "../GTSRB"
    train_data_dir = os.path.join(ROOT_PATH, "Final_Training/Images")
    test_data_dir = os.path.join(ROOT_PATH, "Final_Test/Images")

    images, labels = load_test_data(test_data_dir)
    print("Unique Labels: {0}\nTotal Images: {1}".format(len(set(labels)), len(images)))
    display_one_image(images, labels)
    # display_images_labels(images, labels)
    # display_label_images(images, labels, 16)
    #
    # for image in images[:5]:
    #     print("shape: {0}, min: {1}, max: {2}".format(image.shape, image.min(), image.max()))
    #
    # # Resize images to 32 x 32.
    # images32 = [skimage.transform.resize(image, (32, 32))
    #                 for image in images]
    # display_images_labels(images32, labels)
    #
    # for image in images32[:5]:
    #     print("shape: {0}, min: {1}, max: {2}".format(image.shape, image.min(), image.max()))
