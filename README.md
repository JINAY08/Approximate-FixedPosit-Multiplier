# Approximate-Computing-for-Adversarial-Defense

Contains code for ongoing project

Directory Structure:
- datasets : Datasets used in the project will be stored in this directory
- framework: Contains codebase of Chandan's PyTorch framework
- models   : Trained original models as well as bit-error introduced models used/generated in this project stored in this directory
- results  : The results generated by running various scripts are stored in this folder
- scripts  : Contains scripts for running various experiments, contains script to find accuracy also
- tensors  : Contains tensors of generated adversarial images

## Generating adversarial images

To add adversarial noise to images run the files in scripts directory with following format : {model}\_{dataset}\_{attack}\_tensor.py
For example to add adversarial noise of fgsm attack for resnet18 on cifar10, run the commands below:
```bash
cd scripts/
python3 resnet18_cifar10_fgsm_tensor.py
```
Running these scripts will generate tensors of attacked images and stores them in tensors directory. They will also generate a file named {attack}\_{model}\_{dataset}\_tensor\_result.csv in the results directory. These files will have information of the ground truth, prediction before attack and prediction after attack for all the images. This information is useful to identify teh adversarial images since all attacked images might not be adversarial.

Then to identify adversarial images run
```bash
cd scripts/
python3 adv_img_collector_tensor.py
```
Running this script will generate files named advimgnums_{attack}\_{model}\_{dataset}\_tensor.csv in the results directory. These files contain information only for adversarial images.

## Inferencing on adversarial images with approximate hardware
To generate script to perform inference on adversarial images with approximate designs run the following
```bash
cd framework
python3 scriptmaker.py
```

This will generate a `script.sh` file. Run it using
```bash
cd framework
sh script.sh
```

To compute success rate of attacks, run
```bash
cd scripts/
python3 success_rate_script_new.py
```

# Updated Framework
The following files have been added to the framework:
-> test_posit.c (Updated) : The number of multipliers that are supported has been increased. You could change the multiplier used to multiply dot_pro in the convolution operation (convol function).

->test_posit_compressor: Includes approximate multiplier designs designed using compressors (8x8 unsigned integer). Change the compressor used in the {approx_multiplier} function wherever it is found. (Ctrl-F would be an easy way).

Includes the following compressors:
-Yang1,Yang2,Yang3
-Lin
-Strollo1, Strollo2
-Momeni
-Sabetz
-Venka
-Akbar1, Akbar2
-Ahma
-Ranjbar1, Ranjbar2, Ranjbar3

For changing the operand order, you could swap the order inside {approx_multiply} function, where the {approx_multiplier} is being called.


# Steps for determining accuracy:
-> For generating tensors:

Create a folder named `lenet_mnist` or `resnet18_cifar10` inside the {results} directory based on the network you want to run simulations for. Run the following commands:

```bash
cd scripts/
python3 lenet_mnist_posit.py
```
OR
```bash
cd scripts/
python3 resnet18_cifar10_posit.py
```

This will create the tensors required.
