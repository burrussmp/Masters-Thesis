# AdversarialDefense
Radial Basis Function (RBF) Defense for Neural Networks


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
Note: The parameters of the attack can be adjusted according to IBM's Adversarial Toolbox API which can be modified in the `AdversarialAttacks.py` file to take effect when creating a new attack.
