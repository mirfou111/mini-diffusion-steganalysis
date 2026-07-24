import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm

from src.ddpm import DDPMScheduler
from src.unet import MiniUNet

def train():
    # Configuration Hyperparamètres
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"==> Entraînement sur l'appareil : {device}")

    epochs = 10
    batch_size = 8
    lr = 1e-4
    image_size = 256
    num_timesteps = 1000

    # Transformations : Conversion en niveau de gris + Normalisation [-1, 1]
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # Ramène les pixels de [0, 1] à [-1, 1]
    ])

    # Utilisation d'un dataset factice si data/raw est vide (pour tester le script)
    data_dir = "data/raw"
    if not os.path.exists(data_dir) or len(os.listdir(data_dir)) == 0:
        print("==> Dossier 'data/raw' vide. Génération de données synthétiques de test...")
        os.makedirs(data_dir, exist_ok=True)
        # Création d'un sous-dossier requis par ImageFolder
        dummy_class = os.path.join(data_dir, "images")
        os.makedirs(dummy_class, exist_ok=True)
        from PIL import Image
        for i in range(16):
            img = Image.new('L', (image_size, image_size), color=i*15)
            img.save(os.path.join(dummy_class, f"dummy_{i}.png"))

    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialisation des modules
    scheduler = DDPMScheduler(num_timesteps=num_timesteps)
    model = MiniUNet(in_channels=1, out_channels=1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    print("==> Démarrage de la boucle d'entraînement...")

    for epoch in range(epochs):
        pbar = tqdm(dataloader)
        epoch_loss = 0.0

        for images, _ in pbar:
            images = images.to(device)
            b_size = images.shape[0]

            # 1. Tirer des timesteps aléatoires pour chaque image du batch
            t = torch.randint(0, num_timesteps, (b_size,), device=device).long()

            # 2. Forward Process : Obtenir l'image bruitée x_t et le vrai bruit
            x_t, noise = scheduler.add_noise(images, t)

            # 3. Prédiction du bruit par le UNet
            predicted_noise = model(x_t, t)

            # 4. Calcul de la Loss MSE entre le vrai bruit et le bruit prédit
            loss = criterion(predicted_noise, noise)

            # 5. Rétropropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_description(f"Époque {epoch+1}/{epochs} | Loss: {loss.item():.4f}")

    # Sauvegarde des poids du modèle
    os.makedirs("models", exist_ok=True)
    save_path = "models/ddpm_model.pt"
    torch.save(model.state_dict(), save_path)
    print(f"==> Modèle sauvegardé avec succès dans : {save_path}")

if __name__ == "__main__":
    train()