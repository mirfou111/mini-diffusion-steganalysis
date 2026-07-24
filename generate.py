import os
import torch
from torchvision.utils import save_image

from src.ddpm import DDPMScheduler
from src.unet import MiniUNet

@torch.no_grad()
def sample(num_images=1, output_dir="data/processed"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"==> Génération d'images sur : {device}")

    num_timesteps = 1000
    scheduler = DDPMScheduler(num_timesteps=num_timesteps)
    
    # Rechargement de l'architecture et des poids
    model = MiniUNet(in_channels=1, out_channels=1).to(device)
    model_path = "models/ddpm_model.pt"
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("==> Poids du modèle chargés avec succès.")
    else:
        print("==> Aucun poids trouvé dans 'models/ddpm_model.pt'. Génération avec modèle non entraîné.")

    model.eval()

    # 1. Démarrer d'un bruit pur x_T ~ N(0, I)
    x = torch.randn(num_images, 1, 256, 256, device=device)

    # 2. Boucle de Reverse Process (de t = 999 à t = 0)
    for t_idx in reversed(range(num_timesteps)):
        t = torch.full((num_images,), t_idx, device=device, dtype=torch.long)

        # Prédiction du bruit par le UNet
        predicted_noise = model(x, t)

        # Extraction des paramètres mathématiques du scheduler
        beta_t = scheduler.betas[t_idx].to(device)
        alpha_t = scheduler.alphas[t_idx].to(device)
        alpha_bar_t = scheduler.alphas_cumprod[t_idx].to(device)

        # Ajout de bruit stochastique z si t > 0
        if t_idx > 0:
            noise = torch.randn_like(x)
        else:
            noise = 0

        # Formule du Reverse Step : x_{t-1}
        shape_factor = 1.0 / torch.sqrt(alpha_t)
        noise_factor = beta_t / torch.sqrt(1.0 - alpha_bar_t)
        sigma_t = torch.sqrt(beta_t)

        x = shape_factor * (x - noise_factor * predicted_noise) + sigma_t * noise

    # 3. Denormalisation de [-1, 1] vers [0, 1] pour la sauvegarde
    x = (x + 1.0) / 2.0
    x = torch.clamp(x, 0.0, 1.0)

    # Sauvegarde des images générées
    os.makedirs(output_dir, exist_ok=True)
    for i in range(num_images):
        out_path = os.path.join(output_dir, f"generated_{i+1}.png")
        save_image(x[i], out_path)
        print(f"==> Image synthétique sauvegardée : {out_path}")

if __name__ == "__main__":
    sample(num_images=2)