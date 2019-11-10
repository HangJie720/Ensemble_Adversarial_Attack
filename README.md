# Ensemble Adversarial Black-Box Attacks against Deep Learning Systems
Deep learning (DL) models, e.g., state-of-the-art convolutional neural networks (CNNs), have been widely applied into security sensitivity tasks, such as face payment, security monitoring, automated driving, etc. Then their vulnerability analysis is an emergent topic, especially for black-box attacks, where adversaries do not know the model internal architectures or training parameters. In this paper, two types of ensemble based black-box attack strategies, selective cascade ensemble strategy (SCES) and stack parallel ensemble strategy (SPES), are proposed to explore the vulnerability of DL system and potential factors that contribute to the high-efficiency attacks are explored, SCES adopts a boosting structure of ensemble learning and SPES employs a bagging structure. Moreover, two pairwise and non-pairwise diversity measures are adopted to examine the relationship between the diversity in substitutes ensembles.
In our paper, due to space reasons, some experimental results are not presented, we show these sections as follows:
![image](https://github.com/HangJie720/Ensemble_Adversarial_Attack/blob/master/img/1.png)
                                                        (a) USPS
![image](https://github.com/HangJie720/Ensemble_Adversarial_Attack/blob/master/img/2.png)
                                                        (b) GTSRB
Fig. 1. Transfer rate of adversarial examples crafted by SCES and SPES under different perturbation magnitude α on (a)USPS and (b)GTSRB.
 ![image](https://github.com/HangJie720/Ensemble_Adversarial_Attack/blob/master/img/3.png)
                                     (a) Black-box model trained with adversarial training
 ![image](https://github.com/HangJie720/Ensemble_Adversarial_Attack/blob/master/img/4.png)
                                    (b) Black-box model trained with ensemble adversarial training
Fig. 2. Defense performance of black-box model trained with (a)adversarial training and (b)ensemble adversarial training, against different attacks on MNIST.  
 ![image](https://github.com/HangJie720/Ensemble_Adversarial_Attack/blob/master/img/5.png)
                                     (a) Black-box model trained with adversarial training
 ![image](https://github.com/HangJie720/Ensemble_Adversarial_Attack/blob/master/img/6.png)
                                    (b) Black-box model trained with ensemble adversarial training
Fig. 3. Defense performance of black-box model trained with (a)adversarial training and (b)ensemble adversarial training, against different attacks on GTSRB. 
 ![image](https://github.com/HangJie720/Ensemble_Adversarial_Attack/blob/master/img/7.png)
                                                          (a) SCES
 ![image](https://github.com/HangJie720/Ensemble_Adversarial_Attack/blob/master/img/8.png)
                                                          (b) SPES
Fig. 4. Transfer rate of adversarial examples crafted by (a)SCES and (b)SPES with k=1, 3 and 5 under different perturbation magnitude α on USPS.  
 ![image](https://github.com/HangJie720/Ensemble_Adversarial_Attack/blob/master/img/9.png)
                                                          (a) SCES
 ![image](https://github.com/HangJie720/Ensemble_Adversarial_Attack/blob/master/img/10.png)
                                                          (b) SPES
Fig. 5. Transfer rate of adversarial examples crafted by (a)SCES and (b)SPES with k=1, 3 and 5 under different perturbation magnitude α on MNIST.  


# Delving into Diversity in Substitute Ensembles and Transferability of Adversarial Examples
Our part work has been appeared in "Delving into Diversity in Substitute Ensembles and Transferability of Adversarial Examples", which can be downloaded by: https://link.springer.com/chapter/10.1007/978-3-030-04182-3_16
