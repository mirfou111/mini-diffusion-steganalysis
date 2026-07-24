import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from src.ddpm import DDPMScheduler
from src.unet import MiniUNet

class SimpleImageDataset(Dataset):
    def __init__(self, folder_path, img_size=256):
        self.folder_path = folder_path
        self.files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.pgm'))]
        
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]) # [0, 1] -> [-1, 1]
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("L")
        return self.transform(img)

def train(epochs=50, batch_size=8, lr=2e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"==> Entraînement sur l'appareil : {device}")

    raw_dir = "data/raw"
    if not os.path.exists(raw_dir) or len(os.listdir(raw_dir)) == 0:
        raise ValueError(f"❌ Le dossier '{raw_dir}' est vide ! Exécutez d'abord le script d'extraction.")

    dataset = SimpleImageDataset(raw_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    print(f"📊 Dataset chargé : {len(dataset)} images.")

    scheduler = DDPMScheduler(num_timesteps=1000)
    model = MiniUNet(in_channels=1, out_channels=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    print("==> Démarrage de la boucle d'entraînement...")

    for epoch in range(epochs):
        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Époque {epoch+1}/{epochs}")
        
        for x_0 in pbar:
            x_0 = x_0.to(device)
            b_size = x_0.shape[0]

            # 1. Échantillonnage d'un timestep t aléatoire pour chaque image
            t = torch.randint(0, scheduler.num_timesteps, (b_size,), device=device).long()

            # 2. Bruit gaussien cible
            noise = torch.randn_like(x_0).to(device)

            # 3. Processus Avant (Forward) : Ajout du bruit
            x_t = scheduler.add_noise(x_0, noise, t)

            # 4. Prédiction du bruit par le UNet
            predicted_noise = model(x_t, t)

            # 5. Calcul de la Perte MSE
            loss = criterion(predicted_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

    # Sauvegarde du modèle
    os.makedirs("models", exist_ok=True)
    save_path = "models/ddpm_model.pt"
    torch.save(model.state_dict(), save_path)
    print(f"==> Modèle sauvegardé avec succès dans : {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entraînement DDPM")
    parser.add_argument("--epochs", type=int, default=50, help="Nombre d'époques")
    parser.add_argument("--batch_size", type=int, default=8, help="Taille du batch")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    
    args = parser.parse_args()
    train(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)