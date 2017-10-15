import numpy as np
import tensorflow as tf
import keras.backend as K
from gtsrb import *
from fgs import symbolic_fgs, iter_fgs
from carlini import CarliniLi
from attack_utils import gen_grad
from tf_utils import tf_test_error_rate, batch_eval
from os.path import basename
from sklearn.preprocessing import LabelBinarizer
import os
from tensorflow.python.platform import flags

FLAGS = flags.FLAGS


def main(attack, src_model_name, target_model_names, data_train_dir, data_test_dir):
    np.random.seed(0)
    tf.set_random_seed(0)

    flags.DEFINE_integer('BATCH_SIZE', 32, 'Size of batches')
    set_gtsrb_flags()

    # Get MNIST test data
    _, _, X_test, Y_test = load_data(data_train_dir, data_test_dir)

    # One-hot encode image labels
    label_binarizer = LabelBinarizer()
    Y_test = label_binarizer.fit_transform(Y_test)

    x = tf.placeholder(tf.float32, (None, 32, 32, 1))

    y = tf.placeholder(tf.int32, (None))

    one_hot_y = tf.one_hot(y, 43)

    # source model for crafting adversarial examples
    src_model = load_model(src_model_name)

    # model(s) to target
    target_models = [None] * len(target_model_names)
    for i in range(len(target_model_names)):
        target_models[i] = load_model(target_model_names[i])

    # simply compute test error
    if attack == "test":
        err = tf_test_error_rate(src_model, x, X_test, Y_test)
        print '{}: {:.3f}'.format(basename(src_model_name), err)

        for (name, target_model) in zip(target_model_names, target_models):
            err = tf_test_error_rate(target_model, x, X_test, Y_test)
            print '{}: {:.3f}'.format(basename(name), err)
        return

    eps = args.eps

    # take the random step in the RAND+FGSM
    if attack == "rand_fgs":
        X_test = np.clip(
            X_test + args.alpha * np.sign(np.random.randn(*X_test.shape)),
            0.0, 1.0)
        eps -= args.alpha

    logits = src_model(x)
    grad = gen_grad(x, logits, y)

    # FGSM and RAND+FGSM one-shot attack
    if attack in ["fgs", "rand_fgs"]:
        adv_x = symbolic_fgs(x, grad, eps=eps)

    # iterative FGSM
    if attack == "ifgs":
        adv_x = iter_fgs(src_model, x, y, steps=args.steps, eps=args.eps / args.steps)

    # Carlini & Wagner attack
    if attack == "CW":
        X_test = X_test[0:1000]
        Y_test = Y_test[0:1000]

        cli = CarliniLi(K.get_session(), src_model,
                        targeted=False, confidence=args.kappa, eps=args.eps)

        X_adv = cli.attack(X_test, Y_test)

        r = np.clip(X_adv - X_test, -args.eps, args.eps)
        X_adv = X_test + r

        err = tf_test_error_rate(src_model, x, X_adv, Y_test)
        print '{}->{}: {:.3f}'.format(basename(src_model_name), basename(src_model_name), err)

        for (name, target_model) in zip(target_model_names, target_models):
            err = tf_test_error_rate(target_model, x, X_adv, Y_test)
            print '{}->{}: {:.3f}'.format(basename(src_model_name), basename(name), err)

        return

    if attack == "grad_ens":
        X_test = np.clip(
            X_test + args.alpha * np.sign(np.random.randn(*X_test.shape)),
            0.0, 1.0)
        eps -= args.alpha

        sub_model_ens = (sub_model_1, sub_model_2)
        sub_models = [None] * len(sub_model_ens)
        for i in range(len(sub_model_ens)):
            sub_models[i] = load_model(sub_model_ens[i])

        adv_x = x
        for j in range(args.steps):
            for i, m in enumerate(sub_models + [src_model]):
                logits = m(adv_x)
                gradient = gen_grad(adv_x, logits, y)
                adv_x = symbolic_fgs(adv_x, gradient, eps=args.eps / args.steps, clipping=True)

    # compute the adversarial examples and evaluate
    X_adv = batch_eval([x, y], [adv_x], [X_test, Y_test])[0]

    # white-box attack
    err = tf_test_error_rate(src_model, x, X_adv, Y_test)
    print '{}->{}: {:.3f}'.format(basename(src_model_name), basename(src_model_name), err)

    # black-box attack
    for (name, target_model) in zip(target_model_names, target_models):
        err = tf_test_error_rate(target_model, x, X_adv, Y_test)
        print '{}->{}: {:.3f}'.format(basename(src_model_name), basename(name), err)


if __name__ == "__main__":
    ROOT_PATH = "GTSRB"
    SAVE_PATH = "models"
    train_data_dir = os.path.join(ROOT_PATH, "train.p")
    test_data_dir = os.path.join(ROOT_PATH, "test.p")
    sub_model_1 = os.path.join(SAVE_PATH, "model_sub_1")
    sub_model_2 = os.path.join(SAVE_PATH, "model_sub_2")
    # sub_model_3 = os.path.join(SAVE_PATH, "model_sub_3")
    # sub_model_4 = os.path.join(SAVE_PATH, "model_sub_4")
    # sub_model_5 = os.path.join(SAVE_PATH, "model_sub_5")

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("attack", help="name of attack",
                        choices=["test", "fgs", "ifgs", "rand_fgs", "CW"])
    parser.add_argument("src_model", help="source model for attack")
    parser.add_argument('target_models', nargs='*',
                        help='path to target model(s)')
    parser.add_argument("--eps", type=float, default=0.3,
                        help="FGS attack scale")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="RAND+FGSM random perturbation scale")
    parser.add_argument("--steps", type=int, default=10,
                        help="Iterated FGS steps")
    parser.add_argument("--kappa", type=float, default=100,
                        help="CW attack confidence")

    args = parser.parse_args()
    main(args.attack, args.src_model, args.target_models, train_data_dir, test_data_dir)
