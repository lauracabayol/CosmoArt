import pandas as pd
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms


def select_test_properties(cat, pix_scale):
    """
    Select and process specific galaxy properties for testing.

    This function extracts and processes specific galaxy properties from the provided catalog.
    The properties include disk and bulge effective radii, Sersic indices, ellipticities,
    inclination angles, and bulge fractions. The inclination angles are processed to ensure
    they are within the valid range.

    Args:
        cat (pd.DataFrame): Catalog of galaxy properties.
        pix_scale (float): Pixel scale of the images.

    Returns:
        np.ndarray: Processed array containing selected galaxy properties for testing.
    """

    cat = cat.sample(1)
    disk_res, bulge_res = cat.disk_r50.values / pix_scale, cat.bulge_r50.values / pix_scale
    disk_ns = cat.disk_nsersic.values
    bulge_ns = cat.bulge_nsersic.values
    disk_ellips, bulge_ellips = cat.disk_ellipticity.values, cat.bulge_ellipticity.values
    thetas = cat.inclination_angle.values
    bulge_fracs = cat.bulge_fraction.values

    # Cleaning and processing inclination angle values
    cat.inclination_angle.where(cat.inclination_angle > 0, -cat.inclination_angle, inplace=True)
    cat['inclination_angle'] = cat.inclination_angle / 360 * 2 * np.pi
    thetas = cat.inclination_angle.values
    
    return np.array([bulge_res, disk_res, bulge_ns, disk_ns, bulge_ellips, disk_ellips, thetas, bulge_fracs])[:,0]

def get_features_styleTransfer(image, model, layers=None):
    """ Run an image forward through a model and get the features for 
        a set of layers. Default layers are for VGGNet matching Gatys et al (2016)
    """

    if layers is None:
        layers = {'0': 'conv1_1',
                 '5':  'conv2_1',
                 '10': 'conv3_1',
                 '19': 'conv4_1',
                 '21': 'conv4_2',
                 '28': 'conv5_1'}
        
    features = {}
    x = image
    # model._modules is a dictionary holding each module in the model
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
            
    return features


def gram_matrix(tensor):
    """ Calculate the Gram Matrix of a given tensor for any batch size (b)
        Gram Matrix: https://en.wikipedia.org/wiki/Gramian_matrix
    """
    
    b, d, h, w = tensor.size()
    
    # Reshape the tensor to have batch size as the first dimension
    tensor = tensor.view(b, d, h*w)
    
    # Compute the Gram matrix for each sample in the batch
    gram = torch.bmm(tensor, tensor.transpose(1, 2))
    
    # Normalize the Gram matrix by dividing by the number of elements
    gram = gram / (d * h * w)
    
    return gram


def load_and_resize_image(image_path, size=[150,150]):
    # Open the image using PIL
    image = Image.open(image_path)

    # Define the transformation sequence
    transform = transforms.Compose([
        transforms.Resize((size[0], size[1])),  # Resize the image to 64x64
        transforms.ToTensor(),         # Convert the image to a PyTorch tensor
    ])

    # Apply the transformation to the image
    resized_image = transform(image)

    return resized_image

