import os
import keras
import tensorflow as tf
from keras import backend as K
from tensorflow.python.platform import flags
from keras.models import save_model
from sklearn.preprocessing import LabelBinarizer
from gtsrb import *
from tf_utils import tf_train, tf_test_error_rate
from cleverhans.utils_tf import model_train, model_eval
from attack_utils import gen_grad
from fgs import symbolic_fgs

FLAGS = flags.FLAGS


def main(model_name, adv_model_names, model_type, train_data_dir, test_data_dir):
    np.random.seed(0)
    assert keras.backend.backend() == "tensorflow"
    set_mnist_flags()

    flags.DEFINE_bool('NUM_EPOCHS', args.epochs, 'Number of epochs')

    # Get MNIST test data
    X_train, Y_train, X_test, Y_test = load_data(train_data_dir, test_data_dir)

    # One-hot encode image labels
    label_binarizer = LabelBinarizer()
    Y_train = label_binarizer.fit_transform(Y_train)
    Y_test = label_binarizer.fit_transform(Y_test)

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

    eps = args.eps

    # if src_models is not None, we train on adversarial examples that come
    # from multiple models
    adv_models = [None] * len(adv_model_names)
    for i in range(len(adv_model_names)):
        adv_models[i] = load_model(adv_model_names[i])

    model = model_mnist(type=model_type)
    prediction = model(x)

    x_advs = [None] * (len(adv_models) + 1)

    for i, m in enumerate(adv_models + [model]):
        logits = m(x)
        grad = gen_grad(x, logits, y, loss='training')
        x_advs[i] = symbolic_fgs(x, grad, eps=eps)

    # Train an GTSRB model
    # tf_train(x, y, model, X_train, Y_train, data_gen, x_advs=x_advs)

    train_params = {
        'nb_epochs': args.epochs,
        'batch_size': FLAGS.BATCH_SIZE,
        'learning_rate': 0.001
    }

    def evaluate_1():
        eval_params = {'batch_size': FLAGS.BATCH_SIZE}
        test_accuracy = model_eval(K.get_session(), x, one_hot_y, prediction, X_test, Y_test, args=eval_params)
        print('Test accuracy of modelC on legitimate test '
              'examples: {:.3f}'.format(test_accuracy))

    model_train(K.get_session(), x, one_hot_y, model, X_train, Y_train, data_gen, evaluate=evaluate_1,
                args=train_params)
    # Finally print the result!
    test_error = tf_test_error_rate(model, x, X_test, Y_test)
    print('Test error: %.1f%%' % test_error)
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
    parser.add_argument('adv_models', nargs='*',
                        help='path to adv model(s)')
    parser.add_argument("--type", type=int, help="model type", default=0)
    parser.add_argument("--epochs", type=int, default=12,
                        help="number of epochs")
    parser.add_argument("--eps", type=float, default=0.025,
                        help="FGS attack scale")

    args = parser.parse_args()
    main(args.model, args.adv_models, args.type, train_data_dir, test_data_dir)
