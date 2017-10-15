import numpy as np
import tensorflow as tf
import keras.backend as K
from mnist import set_mnist_flags, load_model
from fgs import symbolic_fgs, iter_fgs
from carlini import CarliniLi
from attack_utils import gen_grad
from tf_utils import tf_test_error_rate, batch_eval
from cleverhans.utils_mnist import data_mnist
from cleverhans.utils_tf import model_eval
from os.path import basename

from tensorflow.python.platform import flags
FLAGS = flags.FLAGS


def main(src_model_name):
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
    src_model = load_model(src_model_name)
    prediction = src_model(x)
    eval_params = {
            'batch_size': FLAGS.BATCH_SIZE
        }

    # error = tf_test_error_rate(src_model, x, X_test, Y_test)
    acc = model_eval(K.get_session(), x, y, prediction, X_test, Y_test, args=eval_params)
    print '{}: {:.3f}'.format(basename(src_model_name), acc)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("src_model", help="source model for attack")

    args = parser.parse_args()
    main(args.src_model)
