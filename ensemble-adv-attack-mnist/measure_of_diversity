import os
import math
import numpy as np
import tensorflow as tf
import keras.backend as K
from mnist import set_mnist_flags, load_model
from tf_utils import tf_test_acc_rate, batch_eval, tf_test_acc_num
from cleverhans.utils_mnist import data_mnist
from cleverhans.utils import display_leg_sample
from os.path import basename
from tensorflow.python.platform import flags
FLAGS = flags.FLAGS


def main(measures, src_model_names):
    np.random.seed(0)
    tf.set_random_seed(0)

    flags.DEFINE_integer('BATCH_SIZE', 32, 'Size of batches')
    set_mnist_flags()

    x = K.placeholder((None,
                       FLAGS.IMAGE_ROWS,
                       FLAGS.IMAGE_COLS,
                       FLAGS.NUM_CHANNELS))

    y = K.placeholder((None, FLAGS.NUM_CLASSES))

    _, _, X_test, Y_test = data_mnist()

    # source model for crafting adversarial examples
    src_models = [None] * len(src_model_names)
    accuracy = [None] * len(src_model_names)
    for i in range(len(src_model_names)):
        src_models[i] = load_model(src_model_names[i])

    if measures == "Q":
        for (name, src_model) in zip(src_model_names, src_models):
            acc = tf_test_acc_rate(src_model, x, X_test, Y_test)
            # print '{}: {:.3f}'.format(basename(name), acc)
            for i in range(len(src_model_names)):
                accuracy[i] = acc
        return

    if measures == "p":
        for (name, src_model) in zip(src_model_names, src_models):
            acc = tf_test_acc_rate(src_model, x, X_test, Y_test)
            # print '{}: {:.3f}'.format(basename(name), acc)
            for i in range(len(src_model_names)):
                accuracy[i] = acc
        return

    if measures == "E":
        lx = 0
        min_sum = 0
        N = len(X_test)
        L = len(src_model_names)

        for j in range(N):

            for (name, src_model) in zip(src_model_names, src_models):
                # the number of substitutes from D that correctly recognize X_test[j]
                lx = lx + tf_test_acc_num(src_model, x, y, X_test[j], Y_test[j])

            min_sum = min_sum + (1.0/(L - math.ceil(L/2.0)))* min(lx, L-lx)
        Ent = (1.0/N) * min_sum
        print('The value of the entropy measures: {:.3f}'.format(Ent))
        return

    if measures == "KW":
        for (name, src_model) in zip(src_model_names, src_models):
            acc = tf_test_acc_rate(src_model, x, X_test, Y_test)
            # print '{}: {:.3f}'.format(basename(name), acc)
            for i in range(len(src_model_names)):
                accuracy[i] = acc
        return


    if measures == "test":
        X_test = X_test[0:5]
        Y_test = Y_test[0:5]
        # display_leg_sample(X_test)
        for j in range(1):

            for (name, src_model) in zip(src_model_names, src_models):
                # the number of substitutes from D that correctly recognize X_test[j]
                num = tf_test_acc_num(src_model, x, y, X_test, Y_test)
                # output 1, 1, 1, 1, 1, 1
                print num


        return

if __name__ == "__main__":
    SAVE_PATH = "models"
    sub_model_1 = os.path.join(SAVE_PATH, "model_sub_1")
    sub_model_2 = os.path.join(SAVE_PATH, "model_sub_2")
    sub_model_3 = os.path.join(SAVE_PATH, "model_sub_3")
    sub_model_4 = os.path.join(SAVE_PATH, "model_sub_4")
    sub_model_5 = os.path.join(SAVE_PATH, "model_sub_5")
    sub_model_6 = os.path.join(SAVE_PATH, "model_sub_6")
    sub_model_7 = os.path.join(SAVE_PATH, "model_sub_7")
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("measures", help="measures of diversity",
                        choices=["Q","p","E","KW","test"])
    parser.add_argument('src_models', nargs='*',
                        help='path to source model(s)')

    args = parser.parse_args()
    main(args.measures, args.src_models)
