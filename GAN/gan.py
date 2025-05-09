import torch.nn as nn
from torch import randn
import torch.nn.utils as nn_utils
class Generator(nn.Module):
    def __init__(self, z_dim=100, img_channels=3, feature_g=256):
        super().__init__()
        def block(in_feat, out_feat,stride=2, padding=1):
            #in_channels, out_channels, kernel_size, stride=1, padding=0
            return nn.Sequential(
                nn_utils.spectral_norm(nn.ConvTranspose2d(in_feat, out_feat, 4, stride, padding)),
                nn.ReLU(True)
            )
            
        self.net = nn.Sequential(
            block(z_dim, feature_g * 8, 1, 0),
            block(feature_g * 8, feature_g * 4, 2, 1),
            block(feature_g * 4, feature_g * 2, 2, 1),
            block(feature_g * 2, feature_g, 2, 1),

            nn.ConvTranspose2d(feature_g, img_channels, 4, 2, 1),  # (bs, 3, 64, 64)
            nn.Tanh()
        )

    def forward(self, z):
        return self.net(z)

class Discriminator(nn.Module):
    def __init__(self, img_channels=3, feature_d=256):
        super().__init__()
        def block(in_feat, out_feat, stride=2, padding=1):
            return nn.Sequential(
                nn_utils.spectral_norm(nn.Conv2d(in_feat, out_feat, 4, stride, padding)),
                nn.LeakyReLU(0.2, inplace=True)
            )
        self.net = nn.Sequential(
            nn.Conv2d(img_channels, feature_d, 4, 2, 1),  # (bs, 64, 32, 32)
            nn.LeakyReLU(0.2, inplace=True),

            block(feature_d, feature_d * 2),
            block(feature_d * 2, feature_d * 4),
            block(feature_d * 4, feature_d * 8),
            
            nn.Conv2d(feature_d * 8, 1, 4, 1, 0),  # (bs, 1, 1, 1)

        )

    def forward(self, x):
        return self.net(x).view(-1)
