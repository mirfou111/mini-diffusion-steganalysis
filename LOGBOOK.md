# Journal d'Exp�rimentation & Notes Acad�miques

## Vue d'ensemble
Ce document retrace la construction pas-�-pas du g�n�rateur DDPM personnalis� et l'analyse de ses artefacts pour le m�moire de st�ganalyse.

## Étape 2 : Le Forward Process (Diffusion Directe)

### 1. Objectif Académique
Modéliser la dégradation progressive d'une image propre $x_0$ en un bruit gaussien pur $x_T$ via une chaîne de Markov à $T$ étapes.

### 2. Équations Clés
- **Echelle de variance linear :** $\beta_t \in [\beta_{\text{start}}, \beta_{\text{end}}]$
- **Facteur de rétention du signal :** $\alpha_t = 1 - \beta_t$ et $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$
- **Formule fermée de saut à $t$ :**
  $$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, \mathbf{I})$$

### 3. Impact sur la Stéganalyse ($H_1$ et $H_2$)
Le paramètre $\bar{\alpha}_t$ montre que pour $t$ élevé, l'information haute fréquence de $x_0$ est progressivement écrasée par la variance du bruit gaussien. C'est ce lissage/remplacement haute fréquence qui crée la rupture statistique mesurée par SRM.

### 4. Intuition & Vulgarisation
Le DDPM repose sur le principe de la d�gradation progressive (Forward) puis de la reconstruction pas-�-pas (Reverse) via un pr�dicteur de bruit (UNet). Les r�sidus de ce d�poussi�rage it�ratif laissent des micro-signatures fr�quentielles responsables des faux positifs SRM.

## Étape 3 : Architecture du Prédicteur de Bruit (UNet & Time Embedding)

### 1. Objectif Académique
Construire le réseau de neurones $\epsilon_\theta(x_t, t)$ capable d'estimer le bruit gaussien $\epsilon$ présent dans une image bruitée $x_t$ à un instant $t$ donné.

### 2. Éléments d'Architecture Clés
1. **Time Embedding (Sinusoidal) :** Injection du timestep $t$ sous forme d'harmoniques de Fourier pour informer le réseau du niveau de bruit à traiter.
2. **Architecture Encodeur-Décodeur (U-Shape) :**
   - **Downsampling :** Compresse spatialement l'image pour capturer le contexte global de la structure d'image.
   - **Skip Connections :** Transfert direct des caractéristiques spatiales haute fréquence de l'encodeur vers le décodeur.
   - **Upsampling :** Reconstruit la résolution $256 \times 256$.

### 3. Impact sur la Stéganalyse ($H_1$ et $H_2$)
Les **Skip Connections** et les opérations de redimensionnement (Up/Downsampling) créent des artefacts périodiques à haute fréquence dans le domaine spatial. Ces motifs (dûs au repliement spectral / aliasing des convolutions) perturbent les filtres SRM.