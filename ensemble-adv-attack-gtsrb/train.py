import os
import keras
import keras.backend as K
from keras.models import save_model
from sklearn.preprocessing import LabelBinarizer
from keras.preprocessing.image import ImageDataGenerator
from tf_utils import tf_train, tf_test_error_rate, tf_test_accuracy_rate
from cleverhans.utils_tf import model_train, model_eval, model_train_advanced
import tensorflow as tf
from tensorflow.python.platform import flags
from gtsrb import *

FLAGS = flags.FLAGS


def main(model_name, model_type, data_train_dir, data_test_dir):
    # K.set_learning_phase(1)
    np.random.seed(0)
    assert keras.backend.backend() == "tensorflow"
    set_gtsrb_flags()

    flags.DEFINE_bool('NUM_EPOCHS', args.epochs, 'Number of epochs')
    # Get MNIST test data
    X_train, Y_train, X_val, Y_val, X_test, Y_test = load_data(data_train_dir, data_test_dir)

    # One-hot encode image labels
    label_binarizer = LabelBinarizer()
    Y_train = label_binarizer.fit_transform(Y_train)
    Y_test = label_binarizer.fit_transform(Y_test)
    Y_val = label_binarizer.fit_transform(Y_val)

    x = tf.placeholder(tf.float32, (None, 32, 32, 1))

    y = tf.placeholder(tf.int32, (None))

    one_hot_y = tf.one_hot(y, 43)

    # Data Augmentation
    data_gen = ImageDataGenerator(featurewise_center=False,
                                  featurewise_std_normalization=False,
                                  width_shift_range=0.1,
                                  height_shift_range=0.1,
                                  zoom_range=0.2,
                                  shear_range=0.1,
                                  rotation_range=10., )

    data_gen.fit(X_train)

    model = model_gtsrb(type=model_type)
    prediction = model(x)

    # Train an GTSRB model
    train_params = {
        'nb_epochs': args.epochs,
        'batch_size': FLAGS.BATCH_SIZE,
        'learning_rate': 0.001
    }

    def evaluate():
        eval_params = {'batch_size': FLAGS.BATCH_SIZE}
        val_accuracy = model_eval(K.get_session(), x, one_hot_y, prediction, X_val, Y_val, args=eval_params)
        print('Validation accuracy of modelA on validation '
              'examples: {:.3f}'.format(val_accuracy))

    def evaluate_2():
        test_accuracy_rate = tf_test_accuracy_rate(model, x, X_val, Y_val)
        print('Validation accuracy rate: %.3f%%' % test_accuracy_rate)

    model_train_advanced(K.get_session(), x, one_hot_y, model, X_train, Y_train, data_gen, evaluate=evaluate_2,
                         args=train_params)

    # Finally print the result!
    test_accuracy_rate = tf_test_accuracy_rate(model, x, X_test, Y_test)
    print('Test accuracy rate: %.3f%%' % test_accuracy_rate)

    # Save model
    save_model(model, model_name)
    json_string = model.to_json()
    with open(model_name + '.json', 'wr') as f:
        f.write(json_string)


if __name__ == '__main__':
    ROOT_PATH = "GTSRB"
    SAVE_PATH = "models"
    train_data_dir = os.path.join(ROOT_PATH, "train.p")
    test_data_dir = os.path.join(ROOT_PATH, "test.p")

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="path to model")
    parser.add_argument("--type", type=int, help="model type", default=1)
    parser.add_argument("--epochs", type=int, default=6, help="number of epochs")
    args = parser.parse_args()

    main(args.model, args.type, train_data_dir, test_data_dir)
