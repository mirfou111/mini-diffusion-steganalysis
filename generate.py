import os
import argparse
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from src.ddpm import DDPMScheduler
from src.unet import MiniUNet

@torch.no_grad()
def generate_images(num_images=5, output_dir="data/generated", model_path="models/ddpm_model.pt"):
    """
    Génère des images synthétiques 256x256 en niveaux de gris à partir du modèle DDPM.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔄 Génération d'images sur : {device}")

    num_timesteps = 1000
    scheduler = DDPMScheduler(num_timesteps=num_timesteps)
    
    model = MiniUNet(in_channels=1, out_channels=1).to(device)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"✅ Poids du modèle chargés avec succès depuis : {model_path}")
    else:
        print(f"⚠️ Aucun poids trouvé à '{model_path}'. Génération à partir du modèle non entraîné.")

    model.eval()

    # 1. Bruit pur x_T ~ N(0, I)
    x = torch.randn(num_images, 1, 256, 256, device=device)

    # 2. Reverse Process (Dépollution progressive)
    print("🎨 Processus de diffusion inverse en cours...")
    for t_idx in tqdm(reversed(range(num_timesteps)), total=num_timesteps, desc="Sampling DDPM"):
        t = torch.full((num_images,), t_idx, device=device, dtype=torch.long)
        predicted_noise = model(x, t)

        beta_t = scheduler.betas[t_idx].to(device)
        alpha_t = scheduler.alphas[t_idx].to(device)
        alpha_bar_t = scheduler.alphas_cumprod[t_idx].to(device)

        noise = torch.randn_like(x) if t_idx > 0 else 0

        shape_factor = 1.0 / torch.sqrt(alpha_t)
        noise_factor = beta_t / torch.sqrt(1.0 - alpha_bar_t)
        sigma_t = torch.sqrt(beta_t)

        x = shape_factor * (x - noise_factor * predicted_noise) + sigma_t * noise

    # 3. Dénormalisation [-1, 1] -> [0, 255] uint8
    x = (x + 1.0) / 2.0
    x = torch.clamp(x, 0.0, 1.0)
    x_np = (x.squeeze(1).cpu().numpy() * 255.0).astype(np.uint8)

    # 4. Sauvegarde des images
    os.makedirs(output_dir, exist_ok=True)
    for i in range(num_images):
        out_path = os.path.join(output_dir, f"synthetic_{i+1:04d}.png")
        img = Image.fromarray(x_np[i], mode='L')
        img.save(out_path)

    print(f"✨ {num_images} images générées avec succès dans le dossier : '{output_dir}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Générateur d'images DDPM")
    parser.add_argument("--num_images", type=int, default=4, help="Nombre d'images à générer")
    parser.add_argument("--out_dir", type=str, default="data/generated", help="Dossier de sortie")
    
    args = parser.parse_args()
    generate_images(num_images=args.num_images, output_dir=args.out_dir)