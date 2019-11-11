import os
import numpy as np
import tensorflow as tf
import keras.backend as K
from mnist import set_mnist_flags, load_model
from fgs import symbolic_fgs, iter_fgs
from carlini import CarliniLi
from attack_utils import gen_grad
from tf_utils import tf_test_error_rate, batch_eval, tf_test_similarity
from cleverhans.utils_mnist import data_mnist
from os.path import basename
from cleverhans.utils import display_leg_adv_sample, save_leg_adv_sample, save_leg_adv_specified_by_user, save_leg_specified_by_user
from tensorflow.python.platform import flags
from usps import load_usps_
FLAGS = flags.FLAGS


def main(attack, src_model_name, target_model_names):
    np.random.seed(0)
    tf.set_random_seed(0)

    # flags.DEFINE_integer('BATCH_SIZE', 32, 'Size of batches')
    set_mnist_flags()

    x = K.placeholder((None,
                       FLAGS.IMAGE_ROWS,
                       FLAGS.IMAGE_COLS,
                       FLAGS.NUM_CHANNELS))

    y = K.placeholder((None, FLAGS.NUM_CLASSES))

    _, _, X_test, Y_test = data_mnist()
    # X_test, Y_test = load_usps_()

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

    if attack == "test_similarity":
        for (name, target_model) in zip(target_model_names, target_models):
            err = tf_test_similarity(src_model, target_model, x, X_test)
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
        X_test = X_test[0:10]
        Y_test = Y_test[0:10]
        cli = CarliniLi(K.get_session(), src_model,
                        targeted=False, confidence=args.kappa, eps=args.eps)

        X_adv = cli.attack(X_test, Y_test)

        r = np.clip(X_adv - X_adv, -args.eps, args.eps)
        X_adv = X_test + r
        np.save('Train_Carlini_100.npy', X_adv)
        np.save('Label_Carlini_100.npy', Y_test)
        err = tf_test_error_rate(src_model, x, X_adv, Y_test)
        print '{}->{}: {:.3f}'.format(basename(src_model_name), basename(src_model_name), err)

        for (name, target_model) in zip(target_model_names, target_models):
            err = tf_test_error_rate(target_model, x, X_adv, Y_test)
            print '{}->{}: {:.3f}'.format(basename(src_model_name), basename(name), err)

        display_leg_adv_sample(X_test, X_adv)
        return

    if attack == "cascade_ensemble":
        # X_test = np.clip(
        #     X_test + args.alpha * np.sign(np.random.randn(*X_test.shape)),
        #     0.0, 1.0)
        # eps_1 = args.eps - args.alpha

        sub_model_ens = (sub_model_2, sub_model_3, sub_model_4, sub_model_5)
        sub_models = [None] * len(sub_model_ens)
        for i in range(len(sub_model_ens)):
            sub_models[i] = load_model(sub_model_ens[i])

        adv_x = x
      
        for i, m in enumerate(sub_models + [src_model]):
            for k in range(args.steps):
                logits = m(adv_x)
                gradient = gen_grad(adv_x, logits, y)
                adv_x = symbolic_fgs(adv_x, gradient, eps=args.eps / args.steps, clipping=True)

    if attack == "Iter_Casc":
        # X_test = np.clip(
        #     X_test + args.alpha * np.sign(np.random.randn(*X_test.shape)),
        #     0.0, 1.0)
        # args.eps = args.eps - args.alpha

        # sub_model_ens = (sub_model_1, sub_model_2, sub_model_3)
        sub_model_ens = (sub_model_1, sub_model_2, sub_model_3, sub_model_4, sub_model_5)
        sub_models = [None] * len(sub_model_ens)
        for i in range(len(sub_model_ens)):
            sub_models[i] = load_model(sub_model_ens[i])

        x_advs = [None] * len(sub_models)
        errs = [None] * len(sub_models)
        adv_x = x
        eps_all = [None] * args.steps

        for i in range(args.steps):
            if i == 0:
                eps_all[0] = (1.0 / len(sub_models)) * args.eps
            else:
                for j in range(i):
                    pre_sum = 0.0
                    pre_sum += eps_all[j]
                    eps_all[i] = (args.eps - pre_sum) * (1.0/len(sub_models))

        for j in range(args.steps):
            print('iterative step is :', j)
            if j == 0:
                for i, m in enumerate(sub_models):
                    logits = m(adv_x)
                    gradient = gen_grad(adv_x, logits, y)
                    adv_x_ = symbolic_fgs(adv_x, gradient, eps=eps_all[j], clipping=True)
                    x_advs[i] = adv_x_

                    X_adv = batch_eval([x, y], [adv_x_], [X_test, Y_test])[0]

                    err = tf_test_error_rate(m, x, X_adv, Y_test)
                    errs[i] = err
                adv_x = x_advs[errs.index(min(errs))]
            else:
                t = errs.index(min(errs))
                print('index of min value of errs:', t)
                logits = sub_models[t](adv_x)
                gradient = gen_grad(adv_x, logits, y)
                adv_x = symbolic_fgs(adv_x, gradient, eps=eps_all[j], clipping=True)

                for i, m in enumerate(sub_models):
                    X_adv = batch_eval([x, y], [adv_x], [X_test, Y_test])[0]
                    err = tf_test_error_rate(m, x, X_adv, Y_test)
                    errs[i] = err
            print('error rate of each substitute models: ', errs)
            print('\t')
            if min(errs) >= 99:
                success_rate = sum(errs) / len(sub_models)
                print('success rate is: {:.3f}'.format(success_rate))
                break

        success_rate = sum(errs) / len(sub_models)
        print('success rate is: {:.3f}'.format(success_rate))

        X_adv = batch_eval([x, y], [adv_x], [X_test, Y_test])[0]
        # np.save('results_mnist/iter_casc_0.1_leg_adv/X_adv_Iter_Casc_0.1.npy', X_adv)

        for (name, target_model) in zip(target_model_names, target_models):
            err = tf_test_error_rate(target_model, x, X_adv, Y_test)
            print '{}->{}: {:.3f}'.format(basename(src_model_name), basename(name), err)

        # save_leg_adv_sample('results_mnist/iter_casc_0.1_leg_adv/', X_test, X_adv)

        # save adversarial example specified by user
        # save_leg_adv_specified_by_user('results_mnist/iter_casc_0.1_leg_adv_label_4/', X_test, X_adv, Y_test)
        save_leg_adv_specified_by_user('results_mnist/Iter_Casc_0.1_leg_adv_label_6_5/', X_test, X_adv, Y_test)
        return

    if attack == "stack_paral":
        # X_test = np.clip(
        #     X_test + args.alpha * np.sign(np.random.randn(*X_test.shape)),
        #     0.0, 1.0)
        # eps -= args.alpha

        sub_model_ens = (sub_model_1, sub_model_2, sub_model_3, sub_model_4, sub_model_5)
        sub_models = [None] * len(sub_model_ens)

        for i in range(len(sub_model_ens)):
            sub_models[i] = load_model(sub_model_ens[i])

        errs = [None] * (len(sub_models) + 1)
        x_advs = [None] * len(sub_models)
        # print x_advs

        eps_all = [None] * args.steps

        for i in range(args.steps):
            if i == 0:
                eps_all[0] = (1.0 / len(sub_models)) * args.eps
            else:
                for j in range(i):
                    pre_sum = 0.0
                    pre_sum += eps_all[j]
                    eps_all[i] = (args.eps - pre_sum) * (1.0 / len(sub_models))

        for i, m in enumerate(sub_models):
            # x = x + args.alpha * np.sign(np.random.randn(*x[0].shape))
            logits = m(x)
            gradient = gen_grad(x, logits, y)
            adv_x = symbolic_fgs(x, gradient, eps=args.eps/3, clipping=True)
            x_advs[i] = adv_x

        # print x_advs
        adv_x_sum = x_advs[0]
        for i in range(len(sub_models)):
            if i == 0: continue
            adv_x_sum = adv_x_sum + x_advs[i]
        adv_x_mean = adv_x_sum / (len(sub_models))
        preds = src_model(adv_x_mean)
        grads = gen_grad(adv_x_mean, preds, y)
        adv_x = symbolic_fgs(adv_x_mean, grads, eps=args.eps, clipping=True)

        # compute the adversarial examples and evaluate
        for i, m in enumerate(sub_models + [src_model]):
            X_adv = batch_eval([x, y], [adv_x], [X_test, Y_test])[0]
            err = tf_test_error_rate(m, x, X_adv, Y_test)
            errs[i] = err

        # compute success rate
        success_rate = sum(errs) / (len(sub_models) + 1)
        print('success rate is: {:.3f}'.format(success_rate))

        # compute transfer rate
        for (name, target_model) in zip(target_model_names, target_models):
            err = tf_test_error_rate(target_model, x, X_adv, Y_test)
            print '{}->{}: {:.3f}'.format(basename(src_model_name), basename(name), err)

        # save adversarial examples
        # np.save('results_mnist/stack_paral_0.1_leg_adv/X_adv_stack_paral_0.1.npy', X_adv)
        # save_leg_adv_sample(X_test, X_adv)
        # save_leg_adv_sample('results_mnist/stack_paral_0.3_leg_adv/', X_test, X_adv)

        # save adversarial example specified by user
        save_leg_adv_specified_by_user('results_mnist/stack_paral_0.1_leg_adv_label_6_5/', X_test, X_adv, Y_test)

        return

    if attack == "parallel_ensemble_Carlini":
        X_test = X_test[0:200]
        Y_test = Y_test[0:200]
        X_adv_1 = np.load('X_adv_1.npy')
        X_adv_2 = np.load('X_adv_2.npy')
        X_adv_3 = np.load('X_adv_3.npy')
        X_adv_4 = np.load('X_adv_4.npy')
        X_adv_5 = np.load('X_adv_5.npy')
        X_adv_mid = (X_adv_1 + X_adv_2 + X_adv_3 + X_adv_4, X_adv_5) / 5
        # x_advs = [None] * 3
        #
        # adv_x_sum = x_advs[0]
        # for i in range(3):
        #     if i==0:continue
        #     adv_x_sum = adv_x_sum + x_advs[i]
        # adv_x_mean  = adv_x_sum / (len(sub_models))

        cli = CarliniLi(K.get_session(), src_model,
                        targeted=False, confidence=args.kappa, eps=args.eps)

        X_adv = cli.attack(X_adv_mid, Y_test)

        r = np.clip(X_adv - X_adv_mid, -args.eps, args.eps)
        X_adv = X_adv_mid + r

        np.save('X_adv_12345.npy', X_adv)
        err = tf_test_error_rate(src_model, x, X_adv, Y_test)
        print '{}->{}: {:.3f}'.format(basename(src_model_name), basename(src_model_name), err)

        for (name, target_model) in zip(target_model_names, target_models):
            err = tf_test_error_rate(target_model, x, X_adv, Y_test)
            print '{}->{}: {:.3f}'.format(basename(src_model_name), basename(name), err)

        display_leg_adv_sample(X_test, X_adv)
        return

    
    # compute the adversarial examples and evaluate
    X_adv = batch_eval([x, y], [adv_x], [X_test, Y_test])[0]
    # np.save('Train-Stack-Paral-5-0.8.npy', X_adv)
    # np.save('Label-Stack-Paral-5-0.8.npy', Y_test)
    # white-box attack
    err = tf_test_error_rate(src_model, x, X_adv, Y_test)
    print '{}->{}: {:.3f}'.format(basename(src_model_name), basename(src_model_name), err)

    # black-box attack
    for (name, target_model) in zip(target_model_names, target_models):
        err = tf_test_error_rate(target_model, x, X_adv, Y_test)
        print '{}->{}: {:.3f}'.format(basename(src_model_name), basename(name), err)

    # save adversarial examples
    # np.save('results/rand_fgs_0.3_leg_adv/X_adv_rand_fgs_0.3.npy', X_adv)
    # display_leg_adv_sample(X_test, X_adv)
    # save_leg_adv_sample('results/rand_fgs_0.3_leg_adv/', X_test, X_adv)

    # save adversarial example specified by user
    save_leg_adv_specified_by_user('results_mnist/Iter_Casc_0.1_leg_adv_label_6_5/', X_test, X_adv, Y_test)
    # save_leg_specified_by_user('results_mnist/label_6/', X_test, Y_test)


if __name__ == "__main__":
    SAVE_PATH = "models"
    RESULT_PATH = "results_mnist"
    sub_model_1 = os.path.join(SAVE_PATH, "model_sub_1")
    sub_model_2 = os.path.join(SAVE_PATH, "model_sub_2")
    sub_model_3 = os.path.join(SAVE_PATH, "model_sub_3")
    sub_model_4 = os.path.join(SAVE_PATH, "model_sub_4")
    sub_model_5 = os.path.join(SAVE_PATH, "model_sub_5")
    sub_model_6 = os.path.join(SAVE_PATH, "model_sub_6")
    sub_model_7 = os.path.join(SAVE_PATH, "model_sub_7")
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("attack", help="name of attack",
                        choices=["test", "test_similarity", "fgs", "ifgs", "rand_fgs", "CW", "cascade_ensemble",
                                 "stack_paral", "parallel_ensemble_Carlini", "Iter_Casc"])
    parser.add_argument("src_model", help="source model for attack")
    parser.add_argument('target_models', nargs='*',
                        help='path to target model(s)')
    parser.add_argument("--eps", type=float, default=0.1,
                        help="FGS attack scale")
    parser.add_argument("--alpha", type=float, default=0.15,
                        help="RAND+FGSM random perturbation scale")
    parser.add_argument("--steps", type=int, default=10,
                        help="Iterated FGS steps")
    parser.add_argument("--kappa", type=float, default=100,
                        help="CW attack confidence")

    args = parser.parse_args()
    main(args.attack, args.src_model, args.target_models)
