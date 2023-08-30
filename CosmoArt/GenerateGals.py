import matplotlib.pyplot as plt
import torch
import numpy as np

def create_fakeGal(model, properties, LatSpace_dim):
    """
    Generate a fake galaxy image using a trained model.

    This function generates a synthetic galaxy image using a trained model by decoding
    a random latent space vector (z) along with the provided properties. The model is
    temporarily set to evaluation mode for inference.

    Args:
        model (nn.Module): Trained model for generating galaxy images.
        properties (list or array-like): Properties associated with the fake galaxy.
        LatSpace_dim (int): Dimensionality of the latent space vector (z).

    Returns:
        np.ndarray: Synthetic galaxy image.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.eval().to(device)
    
    # Generate a random latent space vector (z)
    z = torch.randn(1, LatSpace_dim).to(device)
    
    # Convert properties to a PyTorch tensor
    test_properties = torch.Tensor(properties)
    
    # Decode the latent vector and properties into a synthetic galaxy image
    pred_sample = model.decode(z, test_properties).to(device)
    
    # Detach the tensor from computation graph, move to CPU, and extract the numpy array
    return pred_sample.detach().cpu().numpy()[0, 0]

def view_fakeGal(gal):
    """
    Display a fake galaxy image.

    This function visualizes the synthetic galaxy image using matplotlib.

    Args:
        gal (np.ndarray): Synthetic galaxy image to view.

    Returns:
        None
    """
    plt.figure()
    plt.imshow(gal)
    plt.xlabel('x')
    plt.ylabel('y')
    cbar = plt.colorbar()
    plt.show()
