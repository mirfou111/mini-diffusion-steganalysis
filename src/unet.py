import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SinusoidalPositionEmbeddings(nn.Module):
    """
    Encode l'entier t (timestep) en un vecteur continu via des fonctions sin/cos.
    Permet au réseau de savoir à quelle étape du bruitage il se trouve.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class Block(nn.Module):
    """Bloc de convolution de base intégrant l'information temporelle t."""
    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        # Première convolution
        h = self.relu(self.bn1(self.conv1(x)))
        
        # Injection du Time Embedding
        time_emb = self.relu(self.time_mlp(t_emb))
        h = h + time_emb.unsqueeze(-1).unsqueeze(-1)
        
        # Seconde convolution
        h = self.relu(self.bn2(self.conv2(h)))
        return h


class MiniUNet(nn.Module):
    """
    Architecture UNet simplifiée pour images 256x256 en niveaux de gris (1 canal).
    """
    def __init__(self, in_channels: int = 1, out_channels: int = 1, time_emb_dim: int = 32):
        super().__init__()
        
        # Time Embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )
        
        # Encodeur (Downsampling)
        self.inc = Block(in_channels, 64, time_emb_dim)
        self.down1 = nn.MaxPool2d(2)
        self.down_block1 = Block(64, 128, time_emb_dim)
        self.down2 = nn.MaxPool2d(2)
        self.down_block2 = Block(128, 256, time_emb_dim)

        # Goulot d'étranglement (Bottleneck)
        self.bot = Block(256, 256, time_emb_dim)

        # Décodeur (Upsampling + Skip Connections)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_block2 = Block(256, 128, time_emb_dim)  # 256 car concaténation de 128 + 128
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_block1 = Block(128, 64, time_emb_dim)   # 128 car concaténation de 64 + 64

        # Conv finale pour retrouver 1 canal (bruit prédit)
        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        # 1. Calcul du vecteur temps
        t_emb = self.time_mlp(timestep)

        # 2. Encodeur (Sauvegarde des features pour les Skip Connections)
        x1 = self.inc(x, t_emb)          # [B, 64, 256, 256]
        x2 = self.down_block1(self.down1(x1), t_emb) # [B, 128, 128, 128]
        x3 = self.down_block2(self.down2(x2), t_emb) # [B, 256, 64, 64]

        # 3. Bottleneck
        x_bot = self.bot(x3, t_emb)      # [B, 256, 64, 64]

        # 4. Décodeur + Concaténation des Skip Connections
        x_up2 = self.up2(x_bot)          # [B, 128, 128, 128]
        x_up2 = torch.cat([x_up2, x2], dim=1) # Concaténation [B, 256, 128, 128]
        x_up2 = self.up_block2(x_up2, t_emb)

        x_up1 = self.up1(x_up2)          # [B, 64, 256, 256]
        x_up1 = torch.cat([x_up1, x1], dim=1) # Concaténation [B, 128, 256, 256]
        x_up1 = self.up_block1(x_up1, t_emb)

        # 5. Sortie finale (prédiction du bruit epsilon)
        output = self.outc(x_up1)        # [B, 1, 256, 256]
        return output


if __name__ == "__main__":
    # Test d'intégrité du UNet
    model = MiniUNet(in_channels=1, out_channels=1)
    
    # Simulation d'un batch : 2 images 256x256 bruitées + 2 timesteps
    x_dummy = torch.randn(2, 1, 256, 256)
    t_dummy = torch.tensor([250, 750])
    
    predicted_noise = model(x_dummy, t_dummy)
    
    print("MiniUNet initialisé et testé avec succès !")
    print(f"Forme de l'entrée : {x_dummy.shape}")
    print(f"Forme de la sortie (Bruit prédit) : {predicted_noise.shape}")