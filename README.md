# nox
nox is a convolutional encoder-decoder residual neural network to remove stars from astrophotographs. It uses almost the same architecture as [StarNet](https://github.com/nekitmm/starnet) and was trained using the same technique with adversarial and perceptual losses. The essential difference between StarNet and nox is that nox was trained on a synthetic dataset with artificial but physically realistic stars superimposed onto background images to create training pairs. In contrast, the creator of StarNet created the dataset by manually removing stars from real astrophotographs. nox therefore produces different results to StarNet because it has seen different data during training. I hope that my approach might provide better generalization to diverse optical systems because the training data includes a wide range of star resolution, FWHM, noise, midtones transfer stretch, diffraction spikes, atmospheric conditions (blur and asymmetry), star color, color fringing, etc. Whether I have succeeded, however, remains to be proved by the experience of users!

I am providing here trained weights and the Python code to use them to remove stars from your own images. You can use these directly without having to do any dataset generation or training, although you can do this if you wish. I am also providing the full Python code here to generate your own training datasets, and to train and save a version of nox.

Most of the code was written in Python. Generating the dataset relies on the [Astropy](https://www.astropy.org/) and [Photutils](https://photutils.readthedocs.io/en/stable/api/photutils.datasets.make_model_sources_image.html) Python libraries. I used Tensorflow for machine learning. Other Python libraries are also used, and so you will need to have installed all the relevant packages as per the imports before you can use my code.

An easy-to-use standalone compiled version is in progress. I am just waiting for an [OpenCV bug around importing Tensorflow models with LayerNormalization](https://github.com/opencv/opencv/pull/23882) to be fixed before I can release this.

# Inference - Use provided weights to remove stars from images

```
python nox.py infer input_image.png
```

Note that the input format does not necessarily need to be png.

# Train a version of nox

## Background images
I added artificial stars to images from the [SIDD](https://www.eecs.yorku.ca/~kamel/sidd/dataset.php) and [RENOIR](https://ani.stat.fsu.edu/~abarbu/Renoir.html) datasets (which were originally intended for noise reduction).

1. Download the sRGB images only from "SIDD-Small Dataset". The "Data" directory contains 160 noisy and ground truth image pairs. We are obviously not interested in these pairings, but the different noise levels are useful for generalization, and so all 320 images are included as background images.

1. Download the "aligned" datasets for the three different cameras, which contains 120 noisy and ground truth image pairs. Again, all these images are included in the list of background images.

## Create dataset

1. Ensure that the arguments to the `get_images_paths` function on lines 299 (or 300) point to the downloaded SIDD and RENOIR datasets respectively.

1. Set the desired global parameters on lines 16 - 21. In particular the number of CPUs to use (I used 10), and the number of images to generate (I generated 5000). The performance of the resulting network is closely related to the number of training images.
   
1. Ensure you have installed all the relevant packages as per the imports in `GenerateStars.py`. I recommend use a [`venv` virtual environment](https://docs.python.org/3/library/venv.html) for this, but that is up to you.

1. Open a command prompt and `cd` to the directory containing `GenerateStars.py`. Before running, consider opening a [`screen`](https://www.gnu.org/software/screen/) which will allow you to detach and reattach your terminal window during the long execution time, particularly if you will run it remotely.

1. Run

```
python GenerateStars.py
```

## Training
