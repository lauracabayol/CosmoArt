import torch.nn as nn
import torch
class CondVAE(nn.Module):
    def __init__(self, dim_input, latent_dim=10, size=[150,150]):
        super(CondVAE, self).__init__()
        
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(in_features=dim_input, out_features=size[0]*size[1]),
            nn.Unflatten(1, (1, size[0], size[1], 60)),
            nn.ReLU(),
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        self.fc_mu = nn.Linear(16384, latent_dim)
        self.fc_logvar = nn.Linear(16384, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + 8, 16384), 
            nn.Unflatten(1, (256, 8, 8)),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=2, stride=1, padding=0),
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_logvar(x)

        return mu, log_var
    
    def decode(self, z, properties):
        # Concatenate the sampling (latent distribution) + embedding -> samples conditioned on both the input data and the specified label
        #print(properties.shape, z.shape)
        zcomb = torch.concat((z, properties), 1)
        #print(zcomb.shape)
        
        return self.decoder(zcomb)     
    
    def sampling(self, mu, log_var):
        # calculate standard deviation
        std = log_var.mul(0.5).exp_()
        
        # create noise tensor of same size as std to add to the latent vector
        eps = torch.cuda.FloatTensor(std.size()).normal_()
        
        # multiply eps with std to scale the random noise according to the learned distribution + add combined
        return eps.mul(std).add_(mu) # return z sample 

    def forward(self, x, properties):
        mu, log_var = self.encode(x)
        z = self.sampling(mu, log_var)
        #print(z.shape)

        return self.decode(z, properties), mu, log_var
