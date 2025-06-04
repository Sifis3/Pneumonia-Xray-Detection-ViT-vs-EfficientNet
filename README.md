# Pneumonia Detection from Chest X-Rays Using Vision Transformer and EfficientNet

This repository presents two advanced deep learning pipelines for classifying pneumonia in chest X-ray images using the **Kaggle Chest X-ray Pneumonia Dataset**. The models evaluated include:

- **Vision Transformer (ViT)** using Hugging Face's `TFViTModel`
- **EfficientNetB0** using KerasCV

Both models are trained, evaluated, and interpreted using **Explainable AI (XAI)** techniques, specifically **Grad-CAM**, to visualize which areas of the X-rays influenced each prediction. The goal is to improve diagnostic confidence and model transparency in clinical contexts.

##  Notebooks Included

### 1. `ViT_XAI_Pneumonia_Prediction`
- Implements a Vision Transformer (ViT) pipeline.
- Performs:
  - Data preprocessing and augmentation
  - Model training and validation
  - AUC score: **0.85**
  - Misclassification analysis
  - Grad-CAM heatmaps for interpretability

### 2. `ViT_vs_EfficientNet_Pneumonia_XAI`
- Direct comparison between ViT and EfficientNetB0.
- Applies:
  - Standardized training protocols on both architectures
  - ROC-AUC comparison
  - Grad-CAM overlays for both models
  - Key insight: ViT slightly outperformed EfficientNetB0 in terms of AUC (**ViT ~0.85 vs. EfficientNet ~0.83**) and interpretability clarity.

##  Dataset
**Kaggle Chest X-ray Pneumonia Dataset**  
[https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

Classes:
- NORMAL
- PNEUMONIA

##  Technologies Used
- Python
- TensorFlow / Keras / KerasCV
- Hugging Face Transformers
- Grad-CAM
- Matplotlib, NumPy
- Jupyter Notebook

##  Explainable AI (XAI)
- **Grad-CAM** applied to both architectures.
- Visual heatmaps show model focus on thoracic regions.
- Misclassifications analyzed with visual explanations and possible clinical rationale.

##  Evaluation Metrics
- Accuracy, Precision, Recall, F1-score
- ROC-AUC
- Visual review of misclassified cases

##  Key Insights
- Vision Transformers are competitive with CNNs for medical image classification.
- Grad-CAM helped validate model decisions and highlighted common sources of error.
- Data imbalance and subtle clinical features remain key challenges.

##  How to Run
1. Clone the repository.
2. Ensure you have Python 3.8+, TensorFlow, Hugging Face Transformers, and other dependencies installed.
3. Run each notebook in JupyterLab or VS Code.

##  License
This project is released under the MIT License.

---

*Developed for educational and research purposes. Not for clinical deployment.*
