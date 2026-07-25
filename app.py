import os
import time
import zipfile
import shutil
import psutil
import glob
import torch
import gradio as gr

from train import train
from generate import generate_images

RAW_DIR = "data/raw"
GEN_DIR = "data/generated"
MODEL_PATH = "models/ddpm_model.pt"

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(GEN_DIR, exist_ok=True)
os.makedirs("models", exist_ok=True)


# def get_system_metrics():
#     """Calcul de l'utilisation CPU, RAM et état GPU."""
#     cpu_usage = f"{psutil.cpu_percent()}%"
#     ram_usage = f"{psutil.virtual_memory().percent}%"
    
#     if torch.cuda.is_available():
#         gpu_name = torch.cuda.get_device_name(0)
#         gpu_mem = f"{torch.cuda.memory_allocated(0) / 1024**2:.1f} MB"
#         device_info = f"GPU: {gpu_name} (Utilise: {gpu_mem})"
#     else:
#         device_info = "Processeur (CPU)"
        
#     return f"Appareil: {device_info} | CPU: {cpu_usage} | RAM: {ram_usage}"
def get_system_metrics():
    """Calcul de l'utilisation CPU, RAM et état GPU."""
    cpu_usage = f"{psutil.cpu_percent()}%"
    ram_usage = f"{psutil.virtual_memory().percent}%"  # <-- Parenthèses retirées ici
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = f"{torch.cuda.memory_allocated(0) / 1024**2:.1f} MB"
        device_info = f"GPU: {gpu_name} (Utilise: {gpu_mem})"
    else:
        device_info = "Processeur (CPU)"
        
    return f"Appareil: {device_info} | CPU: {cpu_usage} | RAM: {ram_usage}"


def upload_dataset(file_obj):
    """Importation d'images individuelles ou extraction d'un fichier ZIP."""
    if file_obj is None:
        return "Aucun fichier selectionne.", len(os.listdir(RAW_DIR))
    
    file_path = file_obj.name
    added_count = 0

    if file_path.endswith(".zip"):
        extract_temp = "data/temp_extract"
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_temp)
            
        for root, _, files in os.walk(extract_temp):
            for f in files:
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.pgm')):
                    src = os.path.join(root, f)
                    dst = os.path.join(RAW_DIR, f"custom_{int(time.time())}_{f}")
                    shutil.move(src, dst)
                    added_count += 1
                    
        shutil.rmtree(extract_temp, ignore_errors=True)
    else:
        file_name = os.path.basename(file_path)
        dst = os.path.join(RAW_DIR, f"custom_{int(time.time())}_{file_name}")
        shutil.copy(file_path, dst)
        added_count = 1

    total_images = len(os.listdir(RAW_DIR))
    return f"{added_count} image(s) ajoutee(s) dans '{RAW_DIR}'.", total_images


def run_training(epochs, batch_size, lr, progress=gr.Progress()):
    """Execute la fonction train() avec suivi du temps et metriques."""
    if not os.path.exists(RAW_DIR) or len(os.listdir(RAW_DIR)) == 0:
        return "Erreur: Le dossier 'data/raw' est vide. Veuillez importer des images.", "Echec"

    start_time = time.time()
    device = "CUDA (GPU)" if torch.cuda.is_available() else "CPU"
    
    progress(0, desc="Entrainement en cours...")
    
    try:
        train(epochs=int(epochs), batch_size=int(batch_size), lr=float(lr))
        
        duration = round(time.time() - start_time, 2)
        metrics_summary = (
            f"Entrainement Termine !\n\n"
            f"- Epoques: {epochs}\n"
            f"- Batch Size: {batch_size}\n"
            f"- Learning Rate: {lr}\n"
            f"- Images entrainees: {len(os.listdir(RAW_DIR))}\n"
            f"- Duree totale: {duration} secondes\n"
            f"- Appareil: {device}\n"
            f"- Poids sauvegardes dans: {MODEL_PATH}"
        )
        return metrics_summary, f"Succes ({duration}s)"
    except Exception as e:
        return f"Erreur durant l'entrainement: {str(e)}", "Echec"


def run_generation(num_images, progress=gr.Progress()):
    """Execute generate_images() et mesure la vitesse d'inference."""
    if not os.path.exists(MODEL_PATH):
        return None, f"Erreur: Aucun poids trouve a '{MODEL_PATH}'."

    start_time = time.time()
    progress(0, desc="Diffusion inverse en cours (1000 pas)...")

    generate_images(
        num_images=int(num_images), 
        output_dir=GEN_DIR, 
        model_path=MODEL_PATH
    )

    duration = round(time.time() - start_time, 2)
    time_per_img = round(duration / num_images, 2) if num_images > 0 else 0

    generated_files = sorted(
        glob.glob(os.path.join(GEN_DIR, "*.png")), 
        key=os.path.getmtime, 
        reverse=True
    )[:int(num_images)]
    
    metrics_info = (
        f"Metriques de Generation:\n"
        f"- Images generees: {num_images}\n"
        f"- Pas de diffusion: 1000\n"
        f"- Duree totale: {duration} s\n"
        f"- Vitesse moyenne: {time_per_img} s/image"
    )

    return generated_files, metrics_info


# ==========================================
# INTERFACE GRAPHIQUE GRADIO (BLOCKS)
# ==========================================

with gr.Blocks(title="Mini-Diffusion Steganalysis Studio") as demo:
    
    gr.Markdown("# Mini-Diffusion & Steganalysis Studio")
    gr.Markdown("Interface de gestion des jeux de donnees, d'entrainement DDPM, de generation d'images et d'analyse des performances.")
    
    sys_metrics = gr.Textbox(label="Etat du Systeme", value=get_system_metrics(), interactive=False)
    
    with gr.Tabs():
        
        # TAB 1: DATASET
        with gr.TabItem("1. Gestion du Dataset"):
            gr.Markdown("### Importer des images (PNG, JPG, PGM ou archive ZIP)")
            with gr.Row():
                file_input = gr.File(label="Selectionner un fichier", file_types=[".zip", ".png", ".jpg", ".pgm"])
                upload_btn = gr.Button("Ajouter au Dataset", variant="primary")
            
            upload_status = gr.Markdown()
            count_box = gr.Number(label="Nombre total d'images dans 'data/raw'", value=len(os.listdir(RAW_DIR)), interactive=False)
            
            upload_btn.click(
                fn=upload_dataset, 
                inputs=[file_input], 
                outputs=[upload_status, count_box]
            )

        # TAB 2: ENTRAINEMENT
        with gr.TabItem("2. Entrainement DDPM"):
            gr.Markdown("### Configuration des Hyperparametres")
            with gr.Row():
                epochs_input = gr.Slider(minimum=1, maximum=200, value=50, step=1, label="Epoques")
                batch_input = gr.Slider(minimum=2, maximum=64, value=8, step=2, label="Batch Size")
                lr_input = gr.Dropdown(choices=["1e-3", "2e-4", "1e-4", "5e-5"], value="2e-4", label="Learning Rate")
            
            train_btn = gr.Button("Lancer l'Entrainement", variant="primary")
            
            with gr.Row():
                train_metrics = gr.Markdown()
                status_box = gr.Textbox(label="Statut", value="En attente...", interactive=False)

            train_btn.click(
                fn=run_training, 
                inputs=[epochs_input, batch_input, lr_input], 
                outputs=[train_metrics, status_box]
            )

        # TAB 3: GENERATION
        with gr.TabItem("3. Generation d'Images"):
            gr.Markdown("### Inférence du Modèle")
            num_img_input = gr.Slider(minimum=1, maximum=16, value=4, step=1, label="Nombre d'images a generer")
            gen_btn = gr.Button("Generer les Images", variant="primary")
            
            gen_metrics = gr.Markdown()
            gallery_output = gr.Gallery(label="Images Synthetiques Generees (256x256)", columns=4)

            gen_btn.click(
                fn=run_generation, 
                inputs=[num_img_input], 
                outputs=[gallery_output, gen_metrics]
            )

    demo.load(fn=get_system_metrics, outputs=[sys_metrics])

if __name__ == "__main__":
    demo.queue().launch(share=True)