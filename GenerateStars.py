import matplotlib.pyplot as plt
from matplotlib import interactive
interactive(True)
from astropy.modeling.models import Moffat2D, Gaussian2D
from photutils.datasets import (make_model_sources_image, make_random_models_table, make_gaussian_sources_image,
make_random_gaussians_table, apply_poisson_noise)
import numpy as np
import sys
import pathlib
import os
import glob
import cv2
from threading import Thread
import colorsys

save_dir = os.getcwd() + os.sep + "nox data" + os.sep
cpus = 10
imcount = 5000
start = 0
preview_pairs = False # generate side-by-side image pairs for previewing
color = True

def get_images_paths(root_dir_ssid, root_dir_mi):
    starless_lst = []
    original_lst = []
    
    # SSID dataset image paths
    root = pathlib.Path(root_dir_ssid)
    img_paths = list(root.rglob("*.PNG*"))
    img_paths_lst = [str(path) for path in img_paths]
    for p in img_paths_lst:
        img_type = p.split(os.sep)[-1].split('_')[-3]
        if img_type == "NOISY":
            original_lst.append(p)
            starless_lst.append(p)
        elif img_type == "GT":
            original_lst.append(p)
            starless_lst.append(p)

    # RENOIR dataset image paths
    for p in [x[0] for x in os.walk(root_dir_mi)]:
        noisyImgs = glob.glob(p + os.sep + '*Noisy.bmp')
        original_lst.extend(noisyImgs)
        starless_lst.extend(noisyImgs)
        refImag = glob.glob(p + os.sep + '*Reference.bmp')
        original_lst.extend(refImag)
        starless_lst.extend(refImag)

    original_array = np.asarray(original_lst)
    starless_array = np.asarray(starless_lst)
    return original_array, starless_array
    
def get_random_crop(image, label, crop_height, crop_width):

    max_x = image.shape[1] - crop_width
    max_y = image.shape[0] - crop_height

    x = np.random.randint(0, max_x)
    y = np.random.randint(0, max_y)

    image_crop = image[y: y + crop_height, x: x + crop_width]
    label_crop = label[y: y + crop_height, x: x + crop_width]

    return image_crop, label_crop

def midtones(x, s, m):
    y = np.clip(x, a_min = s, a_max = None) # set all less than s to s
    y = (y - s)/(1. - s); # image from 0 to 1 after clipping
    y = (m - 1.)*y/((2.*m - 1.)*y - m); # midtones transfer function (m = 0.5 for unity)
    return y
    
def add_stars(image, label):
    starModel = Moffat2D()
    shape = image.shape[0:2]

    app = np.random.uniform(low = 0.8, high = 2.2) # arcseconds per pixel, constant per image
    FWHM0 = np.random.triangular(left = 1.8, mode = 2.5, right = 3.2)/app # random FWHM in pixels, constant per image
    m = np.random.uniform(low = 0.01, high = 0.1) # midtones stretch value
    gain = np.random.uniform(low = 0.1, high = 2.)
    starblur = np.random.uniform(low = 0., high = 3.)
    
    # middle-sized stars (large enough for a properly resolved source function, not large enough for bloat) with some midtones transformation
    n_src = 300
    param_ranges = {'amplitude': [1., 1.], 'x_0': [0, shape[1]], 'y_0': [0, shape[0]], 'gamma': [1., 1.], 'alpha': [2.0, 10]} # for range of alpha, see https://academic.oup.com/mnras/article/328/3/977/1247204 ('beta' in reference)
    sources = make_random_models_table(n_src, param_ranges, seed = None)
    sources['amplitude'] = np.multiply(np.power(10., np.random.triangular(left = 0., mode = 0., right = 1., size = n_src)) - 1., 3./9) # replace amplitudes with triangular distribution of magnitudes following trend of local luminosity function, resulting in range 0 to 3
    sources['gamma'] = FWHM0/2./np.sqrt(pow(2., 1./sources['alpha']) - 1.) # calcuate gamma according to alpha and FWHM
    if color:
        sources['kelvins'] = np.random.triangular(left = 800., mode = 800., right = 30000., size = n_src)
        sources['saturation'] = np.random.uniform(low = 0., high = 1., size = n_src)
        hasFringes = np.random.randint(2)
        if hasFringes:
            sources['fringe_band'] = -np.random.uniform(low = 20., high = 30., size = n_src) # vary hue by up to 30 deg either side
            sources['fringe_power'] = np.random.uniform(low = 1., high = 3., size = n_src) # tightness of fringe
        else:
            sources['fringe_band'] = np.zeros(n_src) # constant hue
            sources['fringe_power'] = np.ones(n_src)
        stars1 = make_model_sources_image_color(shape, starModel, sources)
    else: stars1 = make_model_sources_image(shape, starModel, sources)
    stars1 = apply_poisson_noise(stars1*255*gain, seed=None)/255/gain # add Poisson noise
    stars1 = midtones(stars1, 0., m)
    stars1 = cv2.GaussianBlur(stars1, (0, 0), starblur)
    theta0 = np.random.uniform(0, np.pi/2)
    hasSpikes = np.random.randint(2)
    if hasSpikes: stars1 = add_spikes(stars1, sources, theta0)
    
    # large stars - large enough for full-well capacity bleeding / bloating
    n_large = 20
    param_ranges = {'amplitude': [1., 1.], 'x_0': [0, shape[1]], 'y_0': [0, shape[0]], 'gamma': [1., 1.], 'alpha': [2.0, 10]} # for range of alpha, see https://academic.oup.com/mnras/article/328/3/977/1247204 ('beta' in reference)
    sources = make_random_models_table(n_large, param_ranges)
    sources['amplitude'] = np.multiply(np.power(10., np.random.triangular(left = 0., mode = 0., right = 1., size = n_large)) - 1., 1./9) + 1. # replace amplitudes with triangular distribution of magnitudes following trend of local luminosity function, resulting in range 1 to 2
    FWHM = np.random.triangular(left = FWHM0, mode = FWHM0, right = 15., size = n_large) # use range of wide FWHMs to simulate full-well capacity bleeding / bloating
    sources['gamma'] = FWHM/2./np.sqrt(pow(2., 1./sources['alpha']) - 1.) # calcuate gamma according to alpha and FWHM
    if color:
        sources['kelvins'] = np.random.triangular(left = 800., mode = 800., right = 30000., size = n_large)
        sources['saturation'] = np.random.uniform(low = 0., high = 1., size = n_large)
        if hasFringes:
            sources['fringe_band'] = -np.random.uniform(low = 20., high = 30., size = n_large) # vary hue by up to 30 deg either side
            sources['fringe_power'] = np.random.uniform(low = 1., high = 3., size = n_large) # tightness of fringe
        else:
            sources['fringe_band'] = np.zeros(n_large) # constant hue
            sources['fringe_power'] = np.ones(n_large)
        stars2 = make_model_sources_image_color(shape, starModel, sources)
    else: stars2 = make_model_sources_image(shape, starModel, sources)
    stars2 = apply_poisson_noise(stars2*255*gain, seed=None)/255/gain # add Poisson noise
    stars2 = midtones(stars2, 0., m)
    stars2 = cv2.GaussianBlur(stars2, (0, 0), starblur)
    if hasSpikes: stars2 = add_spikes(stars2, sources, theta0)
    
    # small and dim Gaussian stars - subject to a bit more 'wobble'
    n_small = 2500
    stretch_range = 1.1 # range of star 'smearing'
    param_ranges = {'amplitude': [1., 1.], 'x_mean': [0, shape[1]], 'y_mean': [0, shape[0]], 'x_stddev': [1., 1.], 'y_stddev': [1., 1.], 'theta': [0., np.pi]}
    sources = make_random_gaussians_table(n_small, param_ranges, seed = None)
    sources['amplitude'] = np.multiply(np.power(10., np.random.triangular(left = 0., mode = 0., right = 1., size = n_small)) - 1., 0.1/9) # replace amplitudes with triangular distribution of magnitudes following trend of local luminosity function, resulting in range 0 to 0.1
    xstretch = np.random.uniform(low = 1./stretch_range, high = stretch_range) # generate uniform distribution of FWHM x stretch factors
    ystretch = np.random.uniform(low = 1./stretch_range, high = stretch_range) # generate uniform distribution of FWHM x stretch factors
    sources['x_stddev'] = np.multiply(FWHM0/2.35, xstretch) # replace uniform distribution of x stddevs with triangular distribution
    sources['y_stddev'] = np.multiply(FWHM0/2.35, ystretch) # replace uniform distribution of y stddevs with triangular distribution
    if color:
        sources['kelvins'] = np.random.triangular(left = 800., mode = 800., right = 30000., size = n_small)
        sources['saturation'] = np.random.uniform(low = 0., high = 1., size = n_small)
        if hasFringes:
            sources['fringe_band'] = -np.random.uniform(low = 20., high = 30., size = n_small) # vary hue by up to 30 deg either side
            sources['fringe_power'] = np.random.uniform(low = 1., high = 3., size = n_small) # tightness of fringe
        else:
            sources['fringe_band'] = np.zeros(n_small) # constant hue
            sources['fringe_power'] = np.ones(n_small)
        stars3 = make_gaussian_sources_image_color(shape, sources)
    else: stars3 = make_gaussian_sources_image(shape, sources)
    stars3 = apply_poisson_noise(stars3*255*gain, seed=None)/255/gain # add Poisson noise
    stars3 = midtones(stars3, 0., m)
    stars3 = cv2.GaussianBlur(stars3, (0, 0), starblur)
    
    # combine stars to image
    stars = stars1 + stars2 + stars3
    image = 1. - (1. - stars)*(1. - image) # screen blend stars

    image = np.clip(image, 0., 1.)
    return image, label

def add_spikes(stars, sources, theta0): # add simulated diffraction spikes
    FWHM = 2.*sources['gamma']*np.sqrt(pow(2., 1./sources['alpha']) - 1.)
    n_src = len(sources['amplitude'])
    shape = stars.shape[0:2]
    sources['x_mean'] = sources['x_0']
    del sources['x_0']
    sources['y_mean'] = sources['y_0']
    del sources['y_0']
    del sources['gamma']
    del sources['alpha']
    sources['amplitude'] = np.multiply(sources['amplitude'], np.random.uniform(low = 0.8, high = 1.0, size = n_src))
    sources['x_stddev'] = np.multiply(np.multiply(FWHM, sources['amplitude']/15), np.random.uniform(low = 1., high = 1.2, size = n_src))
    sources['y_stddev'] = np.multiply(np.multiply(FWHM, sources['amplitude']), np.random.uniform(low = 1., high = 1.5, size = n_src))
    sources['theta'] = np.ones(n_src)*theta0
    if color:
        sources['fringe'] = np.zeros(n_src) # no color fringing for spikes
        spikes = make_gaussian_sources_image_color(shape, sources)
    else: spikes = make_gaussian_sources_image(shape, sources)
    sources['theta'] = sources['theta'] + np.pi/2
    if color: spikes += make_gaussian_sources_image_color(shape, sources)
    else: spikes += make_gaussian_sources_image(shape, sources)
    spikes = np.clip(spikes, 0., 1.)
    spikes = cv2.GaussianBlur(spikes, (0, 0), np.random.uniform(low = 0.5, high = 1.0))
    return 1. - (1. - stars)*(1. - spikes) # screen blend spikes

def create_images(images, labels, i_from, i_to):
    for i in range(i_from, i_to):
        print('image %d'%i)
        if os.path.exists(save_dir + 'x%d.png'%i) and os.path.exists(save_dir + 'y%d.png'%i): continue
        
        image = cv2.imread(images[i%len(images)])
        label = cv2.imread(labels[i%len(labels)])
        
        if color:
            image = image.astype(np.float32)/255.
            label = label.astype(np.float32)/255.
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)/255.
            label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)/255.
        
        image, label = get_random_crop(image, label, 512, 512)
        image, label = add_stars(image, label)
        
        if color:
            cv2.imwrite(save_dir + 'x%d.png'%i, np.float32(image)*255.)
            cv2.imwrite(save_dir + 'y%d.png'%i, np.float32(label)*255.)
        else:
            cv2.imwrite(save_dir + 'x%d.png'%i, cv2.cvtColor(np.float32(image)*255., cv2.COLOR_GRAY2BGR))
            cv2.imwrite(save_dir + 'y%d.png'%i, cv2.cvtColor(np.float32(label)*255., cv2.COLOR_GRAY2BGR))
        
        if preview_pairs:
            # plot training images
            plt.close()
            fig, ax = plt.subplots(1, 2, sharex = True, figsize=(21, 21))
            if color: ax[0].imshow(np.float32(image))
            else: ax[0].imshow(cv2.cvtColor(np.float32(image), cv2.COLOR_GRAY2BGR))
            ax[0].set_axis_off()
            if color: ax[1].imshow(np.float32(label))
            else: ax[1].imshow(cv2.cvtColor(np.float32(label), cv2.COLOR_GRAY2BGR))
            ax[1].set_axis_off()
            fig.tight_layout()
            plt.savefig(save_dir + '%i.png'%i, bbox_inches = 'tight')
            
def make_model_sources_image_color(shape, model, source_table):
    image = np.zeros((shape[0], shape[1], 3), dtype=float)
    yidx, xidx = np.indices(shape)

    params_to_set = []
    for param in source_table.colnames:
        if param in model.param_names:
            params_to_set.append(param)

    init_params = {param: getattr(model, param) for param in params_to_set}

    try:
        for source in source_table:
            for param in params_to_set:
                setattr(model, param, source[param])
            
            (r, g, b) = kelvin_to_rgb(source['kelvins'])
            (h, l, s) = colorsys.rgb_to_hls(r, g, b)
            L = model(xidx, yidx)
            H = np.ones(xidx.shape, np.uint8)*h*360
            H = H + np.sign(1. - L)*(np.abs(1. - L)**source['fringe_power'])*source['fringe_band'] # color fringing
            S = np.ones(xidx.shape, np.uint8)*source['saturation']
            H = H % 360 # wrap around outside of 0-360 range
            HLS = cv2.merge([H, L, S])
            BGR = cv2.cvtColor(HLS.astype(np.float32), cv2.COLOR_HLS2BGR)
            image += BGR
    finally:
        for param, value in init_params.items():
            setattr(model, param, value)

    return image
    
def make_gaussian_sources_image_color(shape, source_table):
    model = Gaussian2D(x_stddev = 1, y_stddev = 1)

    if 'x_stddev' in source_table.colnames:
        xstd = source_table['x_stddev']
    else:
        xstd = model.x_stddev.value  # default
    if 'y_stddev' in source_table.colnames:
        ystd = source_table['y_stddev']
    else:
        ystd = model.y_stddev.value  # default

    colnames = source_table.colnames
    if 'flux' in colnames and 'amplitude' not in colnames:
        source_table = source_table.copy()
        source_table['amplitude'] = (source_table['flux']/(2.0 * np.pi * xstd * ystd))

    return make_model_sources_image_color(shape, model, source_table)

# From http://www.tannerhelland.com/4435/convert-temperature-rgb-algorithm-code/
# For temperatures in Kelvin in the range 1000 to 40000
def kelvin_to_rgb(kelvin):
    temp = kelvin/100.
    if temp <= 66:
        red = 255.
        green = 99.4708025861*np.log(temp) - 161.1195681661
        if temp <= 19:
            blue = 0.
        else:
            blue = 138.5177312231*np.log(temp - 10.) - 305.0447927307
    else:
        red = 329.698727446*pow(temp - 60., -0.1332047592)
        green = 288.1221695283*pow(temp - 60., -0.0755148492)
        blue = 255.
    
    red = np.minimum(255., np.maximum(0., red))/255.
    blue = np.minimum(255., np.maximum(0., blue))/255.
    green = np.minimum(255., np.maximum(0., green))/255.
    return (red, green, blue)

if __name__ == "__main__":

    images, labels = get_images_paths("../datasets/SIDD_Small_sRGB_Only", "../datasets/RENOIR")
    # images, labels = get_images_paths("C:\\datasets\\SIDD_Small_sRGB_Only", "C:\\datasets\\RENOIR")
    
    # shuffle
    training_data = list(zip(images, labels))
    np.random.shuffle(training_data)
    images, labels = zip(*training_data)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    if not preview_pairs:
        images_per_cpu = int((imcount - start)/cpus)
        threads = []
        for cpu in range(cpus):
            thread = Thread(target = create_images, args = (images, labels, start + cpu*images_per_cpu, start + (cpu + 1)*images_per_cpu))
            threads.append(thread)
            threads[-1].start()
        
        for i in range(len(threads)):
            threads[i].join()
            print('joined thread %i'%i)

    else: create_images(images, labels, 0, imcount)
