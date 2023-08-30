import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader,random_split
from astropy.modeling.models import Sersic2D
import matplotlib.pyplot as plt
from astropy.nddata import block_reduce

class CreateGals():
    
    """
    CreateGals: A class for generating synthetic galaxy images with associated properties.

    This class facilitates the generation of synthetic galaxy images along with their corresponding
    physical properties. It employs the Astropy modeling library to create galaxy profiles based
    on Sersic models, incorporating features such as disk and bulge components, ellipticity,
    inclination angle, and more. The generated galaxy images are normalized and stored alongside
    their associated property vectors.

    Args:
        cat (pd.DataFrame): Catalog of galaxy properties.
        Ngals (int): Number of galaxies to generate.
        size (list): Dimensions of the generated galaxy images.
        pix_scale (float): Pixel scale of the images.

    Attributes:
        training_sample (np.ndarray): Array to store the generated galaxy images.
        properties (np.ndarray): Array to store associated galaxy properties.
        Ngals (int): Number of galaxies to generate.
        size (list): Dimensions of the generated galaxy images.
        cat (pd.DataFrame): Cleaned and sampled galaxy properties catalog.
    """
    
    def __init__(self, cat, Ngals=100, size=[100,100],pix_scale=0.032, res_scale =1):
        
        # Initialization of attributes
        self.training_sample = np.zeros(shape=(Ngals, size[0], size[1]))
        self.properties = np.zeros(shape=(Ngals, 8))
        self.Ngals = Ngals
        self.size = size
        cat = self._clean_catalogue(cat)
        self.cat = cat
        self.res_scale=res_scale

        # Extracting galaxy properties from the catalog
        self.disk_res, self.bulge_res = cat.disk_r50.values / pix_scale, cat.bulge_r50.values / pix_scale
        self.disk_ns = cat.disk_nsersic.values
        self.bulge_ns = cat.bulge_nsersic.values
        self.disk_ellips, self.bulge_ellips = cat.disk_ellipticity.values, cat.bulge_ellipticity.values
        self.thetas = cat.inclination_angle.values
        self.bulge_fracs = cat.bulge_fraction.values

        # Cleaning and processing inclination angle values
        cat.inclination_angle.where(cat.inclination_angle > 0, -cat.inclination_angle, inplace=True)
        cat['inclination_angle'] = cat.inclination_angle / 360 * 2 * np.pi
        self.thetas = cat.inclination_angle.values
        
    def _clean_catalogue(self,cat):
        """
        Clean the provided catalog of galaxy properties.

        Args:
            cat (pd.DataFrame): Uncleaned catalog.

        Returns:
            pd.DataFrame: Cleaned and sampled catalog.
        """
        cat = cat[cat.bulge_r50<3]
        cat = cat.sample(self.Ngals)
        cat = cat.reset_index()
        return cat
        
        
    def create_galaxy(self,i):
        
        """
        Generate a synthetic galaxy image based on provided properties.

        Args:
            i (int): Index of the galaxy in the catalog.

        Returns:
            np.ndarray: Synthetic galaxy image.
        """
        
        # Meshgrid creation
        x, y = np.meshgrid(np.arange(self.size[0]*self.res_scale), np.arange(self.size[1]*self.res_scale))

        # Sersic models for bulge and disk components
        bulge_mod = Sersic2D(amplitude=1, r_eff=self.bulge_res[i]*self.res_scale, n=self.bulge_ns[i],
                             x_0=int(self.size[0]*self.res_scale / 2), y_0=int(self.size[1]*self.res_scale / 2),
                             ellip=self.bulge_ellips[i], theta=0)
        disk_mod = Sersic2D(amplitude=1, r_eff=self.disk_res[i]*self.res_scale, n=self.disk_ns[i],
                            x_0=int(self.size[0]*self.res_scale / 2), y_0=int(self.size[1]*self.res_scale / 2),
                            ellip=self.disk_ellips[i], theta=self.thetas[i])

        # Galaxy profiles
        prof_bulge = bulge_mod(x, y)
        prof_disk = disk_mod(x, y)

        # Amplitudes for bulge and disk components
        bulge_a = self.bulge_fracs[i] / prof_bulge.sum()
        disk_a = (1 - self.bulge_fracs[i]) / prof_disk.sum()

        # Constructing the galaxy image
        if self.disk_res[i] == 0:
            galaxy = bulge_a * prof_bulge
        else:
            galaxy = (bulge_a * prof_bulge) + (disk_a * prof_disk)
            
        galaxy = block_reduce(galaxy, self.res_scale, func=np.mean)  

        return galaxy

    
    def create_training_sample(self):
        """
        Generate a training sample of synthetic galaxy images and associated properties.

        This method iterates over the specified number of galaxies, generating synthetic galaxy
        images using the create_galaxy method. The generated images are normalized by their maximum
        value and stored in the training_sample array. Corresponding galaxy properties are collected
        and stored in the properties array.

        Returns:
            None
        """
        for n in range(self.Ngals):
            gal = self.create_galaxy(n)
            self.training_sample[n] = gal / gal.max()
            self.properties[n] = [self.bulge_res[n], self.disk_res[n], self.bulge_ns[n], 
                                  self.disk_ns[n], self.bulge_ellips[n], self.disk_ellips[n], 
                                  self.thetas[n], self.bulge_fracs[n]]
        return

    def view_gal(self, i):
        """
        Display a synthetic galaxy image.

        This method visualizes the synthetic galaxy image at the specified index using matplotlib.

        Args:
            i (int): Index of the galaxy image to view.

        Returns:
            None
        """
        plt.figure()
        plt.imshow(self.training_sample[i])
        plt.xlabel('x')
        plt.ylabel('y')
        cbar = plt.colorbar()
        plt.show()

    def _to_tensor(self, array):
        """
        Convert a numpy array to a PyTorch tensor.

        Args:
            array (np.ndarray): Numpy array to convert.

        Returns:
            torch.Tensor: Converted PyTorch tensor.
        """
        return torch.Tensor(array)

    def _create_dataset(self, tensor_gals, tensor_prop):
        """
        Create a PyTorch dataset from galaxy images and properties tensors.

        Args:
            tensor_gals (torch.Tensor): Tensor containing galaxy images.
            tensor_prop (torch.Tensor): Tensor containing associated properties.

        Returns:
            TensorDataset: PyTorch dataset containing galaxy images and properties.
        """
        dataset = TensorDataset(tensor_gals, tensor_prop)
        return dataset

    def create_dataloader(self):
        """
        Create a PyTorch DataLoader for the synthetic galaxy dataset.

        This method converts the training_sample and properties arrays to PyTorch tensors,
        creates a dataset using _create_dataset, and then returns a DataLoader for efficient
        batching during training.

        Returns:
            DataLoader: PyTorch DataLoader for the synthetic galaxy dataset.
        """
        gals = self._to_tensor(self.training_sample)
        props = self._to_tensor(self.properties)
        dataset = self._create_dataset(gals, props)
        loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=32, shuffle=True)
        return loader
