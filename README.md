# nox
nox is a convolutional encoder-decoder residual neural network to remove stars from astrophotographs. It uses almost the same architecture as [StarNet](https://github.com/nekitmm/starnet) and was trained using the same technique with adversarial and perceptual losses. The essential difference between StarNet and nox (besides the use of a different normalization technique) is that nox was trained on a synthetic dataset with artificial but physically realistic stars superimposed onto background images to create training pairs. In contrast, the creator of StarNet created the dataset by manually removing stars from real astrophotographs. nox therefore produces different results to StarNet because it has seen different data during training. I hope that my approach might provide better generalization to diverse optical systems because the training data includes a wide range of star resolutions, FWHMs, noise, gain, midtones transfer stretches, diffraction spikes, atmospheric conditions (blurring), star colors, optical aberration (asymmetry, color fringing), etc. Whether I have succeeded, however, remains to be proved by the experience of users!

I am providing [here the trained weights](https://www.dropbox.com/sh/iq7hme4cab3qki7/AACvh3YyAZukJOxp25umtJ_Ra?dl=0) and the Python code to use them to remove stars from your own images. You can use these directly without having to do any dataset generation or training, although you can do this if you wish. I am also providing the full Python code here to generate your own training datasets, and to train a version of nox.

Most of the code was written in Python. Generating the dataset relies on the [Astropy](https://www.astropy.org/) and [Photutils](https://photutils.readthedocs.io/) Python libraries. I used [Tensorflow](https://www.tensorflow.org/) for machine learning. Other Python libraries are also used, and so you will need to have installed all the relevant packages as per the imports before you can use my code.

An easier-to-use standalone compiled version (written in C++) is in progress. I am just waiting for an [OpenCV bug around importing Tensorflow models with LayerNormalization](https://github.com/opencv/opencv/pull/23882) to be fixed before I can release this.

# Inference - Use provided weights to remove stars from images

1. Ensure you have installed all the relevant packages as per the imports in `nox.py`. I recommend using a [`venv` virtual environment](https://docs.python.org/3/library/venv.html) for this, but that is up to you.
1. Download the [trained weights from here](https://www.dropbox.com/sh/iq7hme4cab3qki7/AACvh3YyAZukJOxp25umtJ_Ra?dl=0) and move the `generator_color.h5` and `generator_grayscale.h5` weights files into the same directory as `nox.py`.
1. Set the desired global parameters on lines 26 to 44. These variables cover both inference and training. The relevant ones for inference are:
   - Set `n_channels` to `1` to process grayscale images, or to convert color images to grayscale before processing. Otherwise leave `n_channels` as `3`.
   - The `patch_size` and `stride` variables adjust the size and overlap of each square tile. You can experiment with these if you like.
   - `BATCH_SIZE` is relevant for inference as well as training. It sets the number of samples per batch of computation. If you set this too high, you will run into out-of-memory errors, in which case reduce it, even down to `1`.
1. Run

```
python nox.py infer input_image.png
```

This will create two files, `starry.png` and `starless.png`: `starry.png` is a copy of the input image, and `starless.png` is the result of star removal. Note that the input and output formats do not necessarily need to be png.

# Train a version of nox

## Background images
I added artificial stars to images from the [SIDD](https://www.eecs.yorku.ca/~kamel/sidd/dataset.php) and [RENOIR](https://ani.stat.fsu.edu/~abarbu/Renoir.html) datasets (which were originally intended for noise reduction).

1. Download the sRGB images only from "SIDD-Small Dataset". The "Data" directory contains 160 noisy and ground truth image pairs. We are obviously not interested in these pairings, but the different noise levels are useful for generalization, and so all 320 images are included as background images.
1. Download the "aligned" datasets for the three different cameras, which contains 120 noisy and ground truth image pairs. Again, all these images are included in the list of background images.

Clearly, these datasets are not related to astrophotography in any way. Nevertheless, they seem adequate to train the net to infer some believable background in place of removed stars.

## Create dataset

1. Ensure that the arguments to the `get_images_paths` function on lines 299 (or 300) point to the downloaded SIDD and RENOIR datasets respectively.
1. Set the desired global parameters on lines 16 - 21. In particular `cpus` the number of CPUs to use (I used 10), and `imcount` the number of images to generate (I generated 5000). The performance of the resulting network is closely related to the number of training images. Leave `color` as `True`, even if you want to train a grayscale net, as the conversion from color to grayscale happens during training if necessary. This means you only need to generate one dataset.
1. Ensure you have installed all the relevant packages as per the imports in `GenerateStars.py`. I recommend using a [`venv` virtual environment](https://docs.python.org/3/library/venv.html) for this, but that is up to you.
1. Open a command prompt and `cd` to the directory containing `GenerateStars.py`. Before running, consider opening a [`screen`](https://www.gnu.org/software/screen/) which will allow you to detach and reattach your terminal window during the long execution time, particularly if you will run it remotely.
1. Run

```
python GenerateStars.py
```

The `nox data` folder contains some example training pairs:

![x4.png](/nox%20data/x4.png) ![y4.png](/nox%20data/y4.png)

![x6.png](/nox%20data/x6.png) ![y6.png](/nox%20data/y6.png)

![x13.png](/nox%20data/x13.png) ![y13.png](/nox%20data/y13.png)

![x39.png](/nox%20data/x39.png) ![y39.png](/nox%20data/y39.png)

![x55.png](/nox%20data/x55.png) ![y55.png](/nox%20data/y55.png)

## Training

1. Ensure you have installed all the relevant packages as per the imports in `nox.py`. I recommend using a [`venv` virtual environment](https://docs.python.org/3/library/venv.html) for this, but that is up to you.
1. Set the desired global parameters on lines 26 to 44. These variables cover settings for both inference and training. For training the relevant ones are:
   - `epochs` sets the maximum number of training epochs. I prefer to set this to a large number (`1500`) and kill training with `Ctrl-C` when ready.
   - Set `n_channels` to `1` to train grayscale weights, or to `3` to train color weights. Both use the same color dataset, and conversion to grayscale happens during training, if required.
   - The `patch_size` and `stride` variables adjust the size and overlap of each square tile. You can experiment with these.
   - `BATCH_SIZE` sets the number of samples per batch of computation. If you set this too high, you will run into out-of-memory errors, in which case reduce it, even down to `1`.
   - `ema` sets the decay rate of the exponential moving average. Each epoch, the model's metrics keep `ema` % of the existing state and `(1 - ema)` % of the new state. This has been tuned to produce smooth model metrics that decay fast enough to allow timely assessment of plateauing and learning rate reduction without overtraining.
   - `lr` sets the learning rate
   - `patience` sets the number of epochs with no improvement after which learning rate will be reduced.
   - `cooldown` sets the number of epochs to wait before resuming normal operation after the learning rate has been reduced.
   - `validation` sets whether the dataset should be split into training (80%) and validation (20%) sets. The model will not train on the validation data, and will evaluate the model metrics on this data at the end of each epoch.
   - `save_freq` sets how many training epochs between saving the progress. Every `save_freq` epochs, the model metrics are saved in `history.csv` and plotted in `training.png`. A random image from the validation set is also processed and saved as `input.png`, `output.png` and `gt.png`.
1. Open a command prompt and `cd` to the directory containing `nox.py`. Before running, consider opening a [`screen`](https://www.gnu.org/software/screen/) which will allow you to detach and reattach your terminal window during the long execution time, particularly if you will run it remotely.
1. Run

```
python nox.py train
```

Note that model training can be resumed (including from the same optimizer state) from the files saved every `save_freq` epochs by reissuing the command above.

# License

My code is made available under the MIT License, detailed in the LICENSE.md file inside the repository. Sections of code in nox.py, as indicated by comments, are Copyright (c) 2018-2019 Nikita
Misiura and are used under the same MIT License.
