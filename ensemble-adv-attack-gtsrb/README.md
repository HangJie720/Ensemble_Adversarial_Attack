# Ensemble Adversarial Attack Against Black-Box System Trained by GTSRB Dataset

###### REQUIREMENTS

The code was tested with Python 2.7.13, Tensorflow 1.1.0 and Keras 2.0.5. You also need install tqdm 4.15.0

###### EXPERIMENTS
We start by training a black-box GTSRB model as targeted model to attack.
```
 python -m train models/modelA --type=0 --epochs=30
```
Then we require to train a few simple substitue models used to generate transferable adversarial samples with synthetic inputs by jacobian-based dataset augmentation technique. These are described in _mnist.py_.

```
python -m train models/model_sub_1 --type=1 --epochs=30
python -m train models/model_sub_2 --type=2 --epochs=30
python -m train models/model_sub_3 --type=3 --epochs=30
python -m train models/model_sub_4 --type=4 --epochs=30
python -m train models/model_sub_5 --type=5 --epochs=30
python -m train models/model_sub_6 --type=6 --epochs=30
python -m train models/model_sub_7 --epochs=30  # Select cleverhans.utils_keras substitute() as substitute
python -m train models/model_sub_8 --epochs=30  # Select cleverhans.utils_keras cnn_model() as substitute

```
When you complete substite and black-box target model training, you can make a test to verify the accuracy of trained models.

```
python -m simple_eval test [model(s)]
```

Afterwards, we can use pre-trained substiutes to craft transferable adversarial samples to attack black-box target model to evaluate transferability and robustness to various attacks, we use

```
python -m simple_eval [attack] [source_model] [target_model(s)] [--parameters(opt)]
```
If you select ensemble-based approach to complete black-box attack, you need specify multiple substitutes ensembling to generate transferable adversarial samples. Now, we only consider Gradient-based Ensemble, we will consider Optimization-based Ensemble later.


###### CONTACT
Questions and suggestions can be sent to 1216043136@njupt.edu.cn or 1339327861@qq.com
