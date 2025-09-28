# Facial-Expression-Recognition-System

This project implements a **Facial Expression Recognition System** using the MobileViT deep learning architecture. The system supports **multi‑task facial analysis** for both webcam streams and static images.

**▶ Demo / Videos**  
**Google Drive folder:** https://drive.google.com/drive/folders/1aIiVQed4CYCC6izue8dum36uczN6rn-L?usp=sharing

---

## Tasks
The model performs the following tasks simultaneously (multi‑head outputs):

- **Facial Expression Classification** — 7 classes: `angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprised`  
- **Race Classification** — 5 classes (project-defined taxonomy)  
- **Gender Classification** — 2 classes  
- **Age Estimation** — 10 age groups (project-defined bins)

> Note: The class labels and exact bin definitions should be documented in `data/README.md` (or update the table below).

| Task | #Classes | Example Labels / Notes |
|---|---:|---|
| Expression | 7 | angry, disgust, fear, happy, neutral, sad, surprised |
| Race | 5 | taxonomy per dataset |
| Gender | 2 | female, male (or project-defined) |
| Age | 10 | 0–9, 10–19, 20–29......|

---

## Project Goal
Analyze how **model architecture**, **dataset characteristics**, and **input modality** (grayscale vs. RGB) affect classification accuracy and overall multi‑task performance. We compare MobileViT variants and ablations (e.g., patch size, kernel size, pretraining) to draw conclusions about accuracy/latency trade‑offs and cross‑task interference.

---

## Methods
- **Backbone:** MobileViT (e.g., `mobilevit_xs/s/m`) from `timm`
- **Heads:** Four classification heads on top of a shared backbone
  - Expression (7‑way softmax)
  - Race (5‑way softmax)
  - Gender (2‑way softmax)
  - Age (10‑way softmax)
- **Input Modalities:** RGB vs. grayscale (ablation)
- **Loss:** Weighted sum of cross‑entropy losses for each head
- **Optimization:** AdamW; cosine or step LR scheduler
- **Augmentations:** Random crop/resize, horizontal flip, color jitter (for RGB); keep grayscale consistency where needed
- **Inference:** Real‑time demo on webcam; batch inference for images/videos

## Training
Example CLI (single‑GPU) — adjust paths and hyperparameters as needed:

> To compare **grayscale vs. RGB**, run two experiments with `--modality grayscale` and `--modality rgb`, keeping other settings fixed.

### Multitask Loss (example)
Let CE denote cross‑entropy. The total loss:
\[
\mathcal{L} = \lambda_{exp} \cdot \mathrm{CE}_{exp}
+ \lambda_{race} \cdot \mathrm{CE}_{race}
+ \lambda_{gender} \cdot \mathrm{CE}_{gender}
+ \lambda_{age} \cdot \mathrm{CE}_{age}
\]
Set \(\lambda\) weights to balance tasks (e.g., all 1.0, or tune by validation).

---

## Evaluation

**Metrics**
- Accuracy / F1‑score (per task)
- Confusion matrix per task
- Macro/micro averages for class imbalance
- Real‑time FPS on webcam demo

**Reports**
- CSV metrics (per task + overall)
- Confusion matrices and ROC (where applicable)
- Ablation tables: backbone size, patch/kernel, modality (RGB vs. grayscale)

---

---

## Results (to be filled)
Create a `results/` folder with tables and figures. Example placeholders:

| Setting | Modality | Expr Acc | Race Acc | Gender Acc | Age Acc | FPS |
|---|---|---:|---:|---:|---:|---:|
| MobileViT‑S | RGB |  –  | –  | –  | –  | –  |
| MobileViT‑S | Gray |  –  | –  | –  | –  | –  |
| MobileViT‑XS | RGB |  –  | –  | –  | –  | –  |

---

