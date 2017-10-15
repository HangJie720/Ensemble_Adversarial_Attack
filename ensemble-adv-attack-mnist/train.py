
import keras
import numpy as np
from keras import backend as K
from tensorflow.python.platform import flags
from keras.models import save_model
from cleverhans.utils_tf import model_train, model_eval
from cleverhans.utils_mnist import data_mnist
from cleverhans.utils_keras import cnn_model
from tf_utils import tf_train, tf_test_error_rate
from mnist import model_mnist, set_mnist_flags, data_gen_mnist


FLAGS = flags.FLAGS


def main(model_name, model_type):
    np.random.seed(0)
    assert keras.backend.backend() == "tensorflow"
    set_mnist_flags()
    
    flags.DEFINE_bool('NUM_EPOCHS', args.epochs, 'Number of epochs')

    # Get MNIST test data
    X_train, Y_train, X_test, Y_test = data_mnist()

    # Initialize substitute training set reserved for adversary
    X_sub = X_test[:150]
    Y_sub = np.argmax(Y_test[:150], axis=1)

    # Redefine test set as remaining samples unavailable to adversaries
    X_test = X_test[150:]
    Y_test = Y_test[150:]


    data_gen = data_gen_mnist(X_train)

    x = K.placeholder((None,
                       FLAGS.IMAGE_ROWS,
                       FLAGS.IMAGE_COLS,
                       FLAGS.NUM_CHANNELS
                       ))

    y = K.placeholder(shape=(None, FLAGS.NUM_CLASSES))

    model = model_mnist(type=model_type)
    # model = cnn_model()
    prediction = model(x)
    # Train an MNIST model
    # tf_train(x, y, model, X_train, Y_train, data_gen)

    train_params = {
        'nb_epochs': args.epochs,
        'batch_size': FLAGS.BATCH_SIZE,
        'learning_rate': 0.001
    }

    def evaluate_1():
        eval_params = {'batch_size': FLAGS.BATCH_SIZE}
        test_accuracy = model_eval(K.get_session(), x, y, prediction, X_test, Y_test, args=eval_params)
        print('Test accuracy of blackbox on legitimate test '
              'examples: {:.3f}'.format(test_accuracy))

    model_train(K.get_session(), x, y, model, X_train, Y_train, data_gen, evaluate=evaluate_1, args=train_params)

    save_model(model, model_name)
    json_string = model.to_json()
    with open(model_name+'.json', 'wr') as f:
        f.write(json_string)

    # Finally print the result!
    test_error = tf_test_error_rate(model, x, X_test, Y_test)
    print('Test error: %.1f%%' % test_error)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="path to model")
    parser.add_argument("--type", type=int, help="model type", default=1)
    parser.add_argument("--epochs", type=int, default=30, help="number of epochs")
    args = parser.parse_args()

    main(args.model, args.type)
