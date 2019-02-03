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
from cleverhans.utils import display_leg_adv_sample, display_leg_sample, save_leg_adv_sample, \
    save_leg_adv_specified_by_user

FLAGS = flags.FLAGS


def main(attack, src_model_name, target_model_names, data_train_dir, data_test_dir):
    np.random.seed(0)
    tf.set_random_seed(0)
    set_gtsrb_flags()

    # Get GTSRB test data
    _, _, _, _, X_test, Y_test = load_data(data_train_dir, data_test_dir)

    # display_leg_sample(X_test)

    # One-hot encode image labels
    label_binarizer = LabelBinarizer()
    Y_test = label_binarizer.fit_transform(Y_test)

    x = K.placeholder((None,
                       FLAGS.IMAGE_ROWS,
                       FLAGS.IMAGE_COLS,
                       FLAGS.NUM_CHANNELS))

    y = K.placeholder((None, FLAGS.NUM_CLASSES))

    # one_hot_y = tf.one_hot(y, 43)

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
        X_test = X_test[0:200]
        Y_test = Y_test[0:200]

        cli = CarliniLi(K.get_session(), src_model,
                        targeted=False, confidence=args.kappa, eps=args.eps)

        X_adv = cli.attack(X_test, Y_test)

        r = np.clip(X_adv - X_test, -args.eps, args.eps)
        X_adv = X_test + r
        np.save('Train_Carlini_200.npy', X_adv)
        np.save('Label_Carlini_200.npy', Y_test)

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
        # eps -= args.alpha

        sub_model_ens = (sub_model_2, sub_model_3)
        sub_models = [None] * len(sub_model_ens)
        for i in range(len(sub_model_ens)):
            sub_models[i] = load_model(sub_model_ens[i])

        adv_x = x
        for j in range(args.steps):
            for i, m in enumerate(sub_models + [src_model]):
                logits = m(adv_x)
                gradient = gen_grad(adv_x, logits, y)
                adv_x = symbolic_fgs(adv_x, gradient, eps=args.eps / args.steps, clipping=True)

    if attack == "Iter_Casc":
        # X_test = np.clip(
        #     X_test + args.alpha * np.sign(np.random.randn(*X_test.shape)),
        #     0.0, 1.0)
        # args.eps = args.eps - args.alpha

        sub_model_ens = (sub_model_1, sub_model_2, sub_model_3)
        sub_models = [None] * len(sub_model_ens)
        for i in range(len(sub_model_ens)):
            sub_models[i] = load_model(sub_model_ens[i])

        x_advs = [None] * len(sub_models)
        errs = [None] * len(sub_models)
        adv_x = x
        eps_all = []

        for i in range(args.steps):
            if i == 0:
                eps_all[0] = (1.0 / len(sub_models)) * args.eps
            else:
                for j in range(i):
                    pre_sum = 0.0
                    pre_sum += eps_all[j]
                    eps_all[i] = (args.eps - pre_sum) * (1.0/len(sub_models))

        # for i in range(args.steps):
        #     if i == 0:
        #         eps_0 = (1.0 / len(sub_models)) * args.eps
        #         eps_all.append(eps_0)
        #     elif i == 1:
        #         eps_1 = (1 - 1.0 / len(sub_models)) * (1.0 / len(sub_models)) * args.eps
        #         eps_all.append(eps_1)
        #     elif i == 2:
        #         eps_2 = (1 - (1 - 1.0 / len(sub_models)) * (1.0 / len(sub_models))) * (1.0 / len(sub_models)) * args.eps
        #         eps_all.append(eps_2)
        #     elif i == 3:
        #         eps_3 = (1 - (1 - (1 - 1.0 / len(sub_models)) * (1.0 / len(sub_models))) * (1.0 / len(sub_models))) * (
        #                 1.0 / len(sub_models)) * args.eps
        #         eps_all.append(eps_3)
        #     elif i == 4:
        #         eps_4 = (1 - (
        #                 1 - (1 - (1 - 1.0 / len(sub_models)) * (1.0 / len(sub_models))) * (1.0 / len(sub_models))) * (
        #                          1.0 / len(sub_models))) * (1.0 / len(sub_models)) * args.eps
        #         eps_all.append(eps_4)
        #     elif i == 5:
        #         eps_5 = (1 - (1 - (
        #                 1 - (1 - (1 - 1.0 / len(sub_models)) * (1.0 / len(sub_models))) * (1.0 / len(sub_models))) * (
        #                               1.0 / len(sub_models)))) * (1.0 / len(sub_models)) * args.eps
        #         eps_all.append(eps_5)
        #     elif i == 6:
        #         eps_6 = (1 - (1 - (1 - (
        #                 1 - (1 - (1 - 1.0 / len(sub_models)) * (1.0 / len(sub_models))) * (1.0 / len(sub_models))) * (
        #                                    1.0 / len(sub_models))))) * (1.0 / len(sub_models)) * args.eps
        #         eps_all.append(eps_6)
        #
        #     elif i == 7:
        #         eps_7 = (1 - (1 - (1 - (1 - (
        #                 1 - (1 - (1 - 1.0 / len(sub_models)) * (1.0 / len(sub_models))) * (1.0 / len(sub_models))) * (
        #                                         1.0 / len(sub_models)))))) * (1.0 / len(sub_models)) * args.eps
        #         eps_all.append(eps_7)
        #     elif i == 8:
        #         eps_8 = (1 - (1 - (1 - (1 - (1 - (
        #                 1 - (1 - (1 - 1.0 / len(sub_models)) * (1.0 / len(sub_models))) * (1.0 / len(sub_models))) * (
        #                                              1.0 / len(sub_models))))))) * (1.0 / len(sub_models)) * args.eps
        #         eps_all.append(eps_8)
        #     elif i == 9:
        #         eps_9 = (1 - (1 - (1 - (1 - (1 - (1 - (
        #                 1 - (1 - (1 - 1.0 / len(sub_models)) * (1.0 / len(sub_models))) * (1.0 / len(sub_models))) * (
        #                                                   1.0 / len(sub_models)))))))) * (
        #                         1.0 / len(sub_models)) * args.eps
        #         eps_all.append(eps_9)
        #     elif i == 10:
        #         eps_10 = (1 - (1 - (1 - (1 - (1 - (1 - (1 - (
        #                 1 - (1 - (1 - 1.0 / len(sub_models)) * (1.0 / len(sub_models))) * (1.0 / len(sub_models))) * (
        #                                                         1.0 / len(sub_models))))))))) * (
        #                          1.0 / len(sub_models)) * args.eps
        #         eps_all.append(eps_10)
        #     elif i == 11:
        #         eps_11 = (1 - (1 - (1 - (1 - (1 - (1 - (1 - (1 - (
        #                 1 - (1 - (1 - 1.0 / len(sub_models)) * (1.0 / len(sub_models))) * (1.0 / len(sub_models))) * (
        #                                                              1.0 / len(sub_models)))))))))) * (
        #                          1.0 / len(sub_models)) * args.eps
        #         eps_all.append(eps_11)

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
            print('error rate of each substitute models_oldest: ', errs)
            print('\t')
            if min(errs) >= 99:
                success_rate = sum(errs) / len(sub_models)
                print('success rate is: {:.3f}'.format(success_rate))
                break

        success_rate = sum(errs) / len(sub_models)
        print('success rate is: {:.3f}'.format(success_rate))

        X_adv = batch_eval([x, y], [adv_x], [X_test, Y_test])[0]
        np.save('results/iter_casc_0.2_leg_adv/X_adv_Iter_Casc_0.2.npy', X_adv)

        for (name, target_model) in zip(target_model_names, target_models):
            err = tf_test_error_rate(target_model, x, X_adv, Y_test)
            print '{}->{}: {:.3f}'.format(basename(src_model_name), basename(name), err)

        save_leg_adv_sample('results/iter_casc_0.2_leg_adv/', X_test, X_adv)

        # save adversarial example specified by user
        save_leg_adv_specified_by_user('results/iter_casc_0.2_leg_adv_label_4/', X_test, X_adv, Y_test)
        return

    if attack == "stack_paral":
        # X_test = np.clip(
        #     X_test + args.alpha * np.sign(np.random.randn(*X_test.shape)),
        #     0.0, 1.0)
        # eps -= args.alpha

        sub_model_ens = (sub_model_1, sub_model_2, sub_model_3)
        sub_models = [None] * len(sub_model_ens)

        for i in range(len(sub_model_ens)):
            sub_models[i] = load_model(sub_model_ens[i])

        errs = [None] * (len(sub_models) + 1)
        x_advs = [None] * len(sub_models)
        # print x_advs

        for i, m in enumerate(sub_models):
            # x = x + args.alpha * np.sign(np.random.randn(*x[0].shape))
            logits = m(x)
            gradient = gen_grad(x, logits, y)
            adv_x = symbolic_fgs(x, gradient, eps=args.eps / 2, clipping=True)
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
        np.save('results/stack_paral_0.2_leg_adv/X_adv_stack_paral_0.2.npy', X_adv)
        # save_leg_adv_sample(X_test, X_adv)
        save_leg_adv_sample('results/stack_paral_0.2_leg_adv/', X_test, X_adv)

        # save adversarial example specified by user
        save_leg_adv_specified_by_user('results/stack_paral_0.2_leg_adv_label_4/', X_test, X_adv, Y_test)

        return

    if attack == "cascade_ensemble_2":
        X_test = np.clip(
            X_test + args.alpha * np.sign(np.random.randn(*X_test.shape)),
            0.0, 1.0)
        eps -= args.alpha

        sub_model_ens = (sub_model_1, sub_model_2)
        sub_models = [None] * len(sub_model_ens)

        for i in range(len(sub_model_ens)):
            sub_models[i] = load_model(sub_model_ens[i])

        x_advs = [([None] * len(sub_models)) for i in range(args.steps)]
        # print x_advs

        x_adv = x
        for j in range(args.steps):
            for i, m in enumerate(sub_models):
                logits = m(x_adv)
                gradient = gen_grad(x_adv, logits, y)
                x_adv = symbolic_fgs(x_adv, gradient, eps=args.eps / args.steps, clipping=True)
                x_advs[j][i] = x_adv

        # print x_advs
        adv_x_sum = x_advs[0][0]
        for j in range(args.steps):
            for i in range(len(sub_models)):
                if j == 0 and i == 0: continue
                adv_x_sum = adv_x_sum + x_advs[j][i]
        adv_x_mean = adv_x_sum / (args.steps * len(sub_models))
        preds = src_model(adv_x_mean)
        grads = gen_grad(adv_x_mean, preds, y)
        adv_x = symbolic_fgs(adv_x_mean, grads, eps=args.eps / args.steps, clipping=True)

    # compute the adversarial examples and evaluate
    X_adv = batch_eval([x, y], [adv_x], [X_test, Y_test])[0]

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
    # save_leg_adv_specified_by_user('results/rand_fgs_0.1_leg_adv_label_4/', X_test, X_adv, Y_test)


if __name__ == "__main__":
    ROOT_PATH = "GTSRB"
    SAVE_PATH = "models_oldest"
    train_data_dir = os.path.join(ROOT_PATH, "train.p")
    test_data_dir = os.path.join(ROOT_PATH, "test.p")
    sub_model_1 = os.path.join(SAVE_PATH, "modelB")
    sub_model_2 = os.path.join(SAVE_PATH, "modelC")
    sub_model_3 = os.path.join(SAVE_PATH, "modelD")
    sub_model_4 = os.path.join(SAVE_PATH, "modelA_sub_1")
    sub_model_5 = os.path.join(SAVE_PATH, "modelA_sub_2")
    sub_model_6 = os.path.join(SAVE_PATH, "modelA_sub_3")
    sub_model_7 = os.path.join(SAVE_PATH, "modelA_sub_7")

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("attack", help="name of attack",
                        choices=["test", "fgs", "ifgs", "rand_fgs", "CW", "cascade_ensemble", "Stack_Paral",
                                 "Iter_Casc"])
    parser.add_argument("src_model", help="source model for attack")
    parser.add_argument('target_models', nargs='*',
                        help='path to target model(s)')
    parser.add_argument("--eps", type=float, default=0.3,
                        help="FGS attack scale")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="RAND+FGSM random perturbation scale")
    parser.add_argument("--steps", type=int, default=3,
                        help="Iterated FGS steps")
    parser.add_argument("--kappa", type=float, default=100,
                        help="CW attack confidence")

    args = parser.parse_args()
    main(args.attack, args.src_model, args.target_models, train_data_dir, test_data_dir)
