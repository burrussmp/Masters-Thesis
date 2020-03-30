# Master's Thesis: "Enhancing the Robustness of Deep Neural Networks Using Radial Basis Functions"
## Author: Matthew P. Burruss
## Date: May 8th 2020

Enhancing the robustness of deep neural networks using radial basis functions (RBFs). The [paper can be found here](https://www.linkedin.com/posts/matthew-burruss-6034a2126_masters-thesis-activity-6646062841801555968-RWdl).

## Replicating the physical attack RBF detection mechanism performed on DeepNNCar
Please see [this google colab notebook](https://drive.google.com/open?id=1GHh84ECYNfhruTaf9eU5CWyvN4N5qr1b)

For videos of DeepNNCar using the RBF to detect the physical attack in real-time, videos can be found [here](https://drive.google.com/drive/folders/10Ek4SH2mBVL-M8pUb7pH-dT_qGDcblDs).

## Replicating the data poisoning attacks and the RBF outlier detection method to clean poisoned data sets.
Please see [this google colab notebook](https://drive.google.com/open?id=1YK2ROlEGKfAv6QWBhfs_mGmEgNQa7xfv)

## Replicating InceptionV3 Data
1. Download the following
[Link to 10 imagenet classes](https://drive.google.com/drive/folders/1gaVBGbIA7qOOq8y1cRXq21tmsmR0LJ4-?usp=sharing)
[Link to model weights (.h5)](https://drive.google.com/drive/folders/1FpTgA-2DbsvMHhPhx3TwKatU52BsxkR9?usp=sharing)
[Pre-Computed Adversarial Images](https://drive.google.com/drive/folders/1eQqUaIa84bIZ8F4I_yCO5DY4hceLyEZ6?usp=sharing)

2. ```git clone https://github.com/burrussmp/AdversarialDefense & cd AdversarialDefense/src/```

3. Edit `InceptionV3_Attack.py`

- change `baseDir` to path pointing to folder containing model weights
- change `imagenet_baseDir` to path pointing to folder containing link to 10 imagenet classes
- change `attackBaseDir` to path pointing pointing to AdversaryAttacks folder

### If not re-creating new images (Easiest Option)
1. Run `python3 InceptionV3_Attack.py`

### If creating new images using the adversarial algorithms (Difficult Option)
This is slightly messy because each attack was not performed at the same time so there are multiple "clean" x and y npy files. For example, './AdversarialAttack/Attack_FGSM_IFGSM_DeepFool' contains x_test_adv_orig.npy and y_test_adv_orig.npy which are the original, clean versions of the adversarial images found in the subdirectories deepfool, fgsm, and ifgsm.

1. Comment `evaluateAttack(...)` and uncomment `createAttack(...)` Line 306 and Line 307 respectively

2. Running `python3 InceptionV3_Attack.py` will create the following file structure based on `attackBaseDir` (Note this can take some time!)

```
attackBaseDir ---|
                 |
                 x_test_adv_orig.npy
                 y_test_adv.orig.npy
                 fgsm/ -------|
                             |
                             anomaly_clean_attack.npy
                             softmax_clean.npy
                 ifgsm/ -------|
                             |
                             anomaly_clean_attack.npy
                             softmax_clean.npy
                 deepfool/ -------|
                             |
                             anomaly_clean_attack.npy
                             softmax_clean.npy
                 c&w/ -------|
                             |
                             anomaly_clean_attack.npy
                             softmax_clean.npy
                 fgsm/ -------|
                             |
                             anomaly_clean_attack.npy
                             softmax_clean.npy
                 pgd/ -------|
                             |
                             anomaly_clean_attack.npy
                             softmax_clean.npy
```
3. After all the attacks conclude, change the `path` for every field of an attack in `evaluate_attack()` to simply `attackBaseDir`
## IMPORTANT
If you would like to perform your own tests using the trained InceptionV3 or would like to use the pre-trained model in a transfer learning scenario, use the preprocess function in `InceptionV3_Attack.py`. Thanks!

Note: The parameters of the attack can be adjusted according to IBM's Adversarial Toolbox API which can be modified in the `AdversarialAttacks.py` file to take effect when creating a new attack.
