# Glaucoma Grading with Single and Multi-Modality Deep Learning  
*Fundus Photography (CFP) + Optical Coherence Tomography (OCT)*  

This repository contains the official implementation of our URECA@NTU 2024–25 research project:  

**Single and Multi-Modality Fusion Deep Learning Approaches for Glaucoma Grading Using Fundus and OCT Imaging**  

---

## 🔍 Overview  
Glaucoma is a leading cause of irreversible blindness. Automated grading of its severity from retinal imaging is critical for clinical decision-making.  

In this work, we investigate:  
- **Single-modality approaches** using **RETFound**, a transformer-based foundation model, fine-tuned on:  
  - **Fundus (CFP)** images (REFUGE2 + GAMMA datasets)  
  - **OCT slices** (GAMMA dataset)  
- **Multi-modality approaches** combining **CFP + OCT** using:  
  - **Dual-branch ResNet34** (CNN-based fusion)  
  - **Dual-branch RETFound ViT** (Transformer-based fusion)  

Our experiments show that while RETFound excels in single-modality settings, the **dual-branch CNN fusion model** achieves the most reliable multi-class glaucoma grading.  

---

## 📊 Key Results  

| Model | Modality | Accuracy | Kappa | F1-score | Notes |
|-------|----------|----------|-------|----------|-------|
| RETFound (ViT-L) | CFP (REFUGE2) | ~90% | 0.67 | 0.94 | High sensitivity, moderate specificity |
| RETFound (ViT-L) | CFP (GAMMA) | ~90% | 0.79 | 0.92 | Balanced performance |
| RETFound (ViT-L) | OCT slices | ~78% | 0.64 | 0.80 | Very high sensitivity, low specificity |
| Dual-branch ResNet34 | CFP + OCT | **80%** | **0.875** | **0.77** | Best multi-modal performance |
| Dual-branch RETFound ViT | CFP + OCT | 60% | 0.37 | 0.57 | Struggled with small dataset |

---
