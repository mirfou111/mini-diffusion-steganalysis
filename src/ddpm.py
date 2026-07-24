import torch
import torch.nn as nn

class DDPMScheduler:
    """
    Gestionnaire du Forward Process (bruitage) pour le modèle DDPM.
    Implémente la dégradation fermée d'une image x_0 vers un état x_t.
    """
    def __init__(self, num_timesteps: int = 1000, beta_start: float = 0.0001, beta_end: float = 0.02):
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end

        # Schedule linéaire des variances beta_t
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        
        # Alpha_t = 1 - beta_t (proportion d'image conservée)
        self.alphas = 1.0 - self.betas
        
        # Alpha_bar_t = produit cumulé des alphas (facteur global de signal à l'étape t)
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    # def add_noise(self, x_0: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    #     """
    #     Applique le Forward Process en une seule étape (Reparametrization Trick).
        
    #     Args:
    #         x_0 (torch.Tensor): Image originale propre [B, C, H, W]
    #         t (torch.Tensor): Timesteps cibles pour chaque image du batch [B]
            
    #     Returns:
    #         tuple[torch.Tensor, torch.Tensor]: (x_t image bruitée, epsilon bruit ajouté)
    #     """
    #     # Génération du bruit gaussien pur epsilon ~ N(0, I)
    #     noise = torch.randn_like(x_0)
        
    #     # Récupération de sqrt(alpha_bar_t) et sqrt(1 - alpha_bar_t) pour le batch
    #     sqrt_alpha_bar = torch.sqrt(self.alphas_cumprod[t]).view(-1, 1, 1, 1)
    #     sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - self.alphas_cumprod[t]).view(-1, 1, 1, 1)
        
    #     # Formule directe : x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
    #     x_t = sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise
        
    #     return x_t, noise
    def add_noise(self, x_0: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Applique le Forward Process en une seule étape (Reparametrization Trick).
        """
        # S'assurer que alphas_cumprod est sur le même device (CPU/CUDA) que x_0
        self.alphas_cumprod = self.alphas_cumprod.to(x_0.device)
        
        noise = torch.randn_like(x_0)
        
        sqrt_alpha_bar = torch.sqrt(self.alphas_cumprod[t]).view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - self.alphas_cumprod[t]).view(-1, 1, 1, 1)
        
        x_t = sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise
        
        return x_t, noise


if __name__ == "__main__":
    # Test d'intégrité du scheduler
    scheduler = DDPMScheduler(num_timesteps=1000)
    dummy_image = torch.randn(2, 1, 256, 256)  # Batch de 2 images 256x256 en niveaux de gris
    timesteps = torch.tensor([100, 800])        # Étape 100 et 800
    
    noisy_img, noise_added = scheduler.add_noise(dummy_image, timesteps)
    print("✅ DDPM Scheduler initialisé avec succès !")
    print(f"Forme de l'image bruitée : {noisy_img.shape}")