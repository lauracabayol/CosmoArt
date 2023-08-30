import tqdm
import torch
import torch.nn as nn

def condvae_loss(recon_x, x, mu, logvar):
    """
    Calculate the conditional Variational Autoencoder (cVAE) loss.

    This function computes the cVAE loss, which consists of two components:
    - Reconstruction loss: Measures the discrepancy between the reconstructed
      data and the original input.
    - KL divergence loss: Quantifies the difference between the learned latent
      distribution and the desired prior distribution (Gaussian).

    Args:
        recon_x (torch.Tensor): Reconstructed data from the VAE.
        x (torch.Tensor): Original input data.
        mu (torch.Tensor): Latent variable mean.
        logvar (torch.Tensor): Logarithm of latent variable variance.

    Returns:
        torch.Tensor: Computed cVAE loss.
    """
    
    # binary cross-entropy loss element-wise and sums up the individual losses
    reconstruction_loss = nn.BCELoss(reduction='sum')(recon_x, x)
    
    # quantifies the difference between the learned latent distribution and the desired prior distribution (Gaussian)
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return reconstruction_loss + kl_divergence

def VAE_trainEpoch(model, optimizer, train_loader, dim_in=100):
    """
    Train a Variational Autoencoder (VAE) for one epoch.

    This function trains a VAE for one epoch using the provided data loader.
    It calculates the cVAE loss, performs backpropagation, and updates the model's parameters.

    Args:
        model (nn.Module): VAE model to be trained.
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        train_loader (DataLoader): DataLoader containing training data.
        dim_in (int): Dimensionality of the input noise.

    Returns:
        float: Average loss for the epoch.
    """
    model = model.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_loss = 0

    progress_bar = tqdm.tqdm(train_loader, desc="Epoch Progress", leave=False)
    for data, properties in progress_bar:
        properties = properties.to(device)
        datain = torch.randn(size=(len(properties), dim_in)).to(device)
        data = data.unsqueeze(1).cuda()
        optimizer.zero_grad()

        recon_batch, mu, log_var = model(datain, properties)
        loss = condvae_loss(recon_batch, data, mu, log_var)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix({"Loss": total_loss / (progress_bar.n + 1)})

    return total_loss / len(train_loader)

def VAE_train(model, optimizer, train_loader, epochs, save_path=None):
    """
    Train a Variational Autoencoder (VAE) for multiple epochs.

    This function trains a VAE for the specified number of epochs using the provided data loader.
    It prints the epoch progress and the computed loss for each epoch.

    Args:
        model (nn.Module): VAE model to be trained.
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        train_loader (DataLoader): DataLoader containing training data.
        epochs (int): Number of epochs for training.

    Returns:
        None
    """
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        epoch_loss = VAE_trainEpoch(model, optimizer, train_loader)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")
        
    if save_path!=None:
        torch.save(model, save_path)

