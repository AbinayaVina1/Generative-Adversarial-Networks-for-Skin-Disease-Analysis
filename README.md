# Generative-Adversarial-Networks-for-Skin-Disease-Analysis
## 🎯 Overview
This project addresses the critical challenge of **limited annotated medical imaging data** by
leveraging advanced deep learning techniques to generate high-quality synthetic Diabetic Foot
Ulcer (DFU) images. Our hybrid framework combines **Wasserstein GAN with Gradient Penalty
(WGAN-GP)** and **Long Short-Term Memory (LSTM)** networks to produce realistic medical
images that enhance disease classification accuracy and support automated diagnosis systems.
### 🔬 Research Highlights
- **87% Classification Accuracy** with 0.93 AUROC on synthetic data
- **Inception Score: 7.10** | **FID: 28** | **PSNR: 35 dB** | **SSIM: 0.85**
- Superior performance compared to EfficientNetV2-based variants
- Clinically validated synthetic image generation
- Real-world infection detection with >0.94 confidence for severe cases
---
## ✨ Key Features
### 🎨 Synthetic Image Generation
- High-fidelity DFU image synthesis using WGAN-GP
- Temporal dependency modeling with 2-layer LSTM
- Addresses class imbalance and data scarcity issues
- Privacy-preserving dataset augmentation
### 🤖 Automated Diagnosis
- Binary classification (Infected vs Non-infected)
- Three hybrid architectures: WGAN-GP+LSTM, EfficientNetV2M+LSTM,
EfficientNetV2S+LSTM
- Confidence calibration for clinical reliability
- Real-time inference capabilities
### 📊 Comprehensive Evaluation
- Multiple quality metrics (IS, FID, PSNR, SSIM)
- Clinical performance metrics (Accuracy, Precision, Recall, F1-Score, AUROC)
- Benchmarked against state-of-the-art methods
---
## ️ Architecture
### WGAN-GP + LSTM Generator
```
┌───────────────────────────────────────────────────────────
──┐
│ Latent Vector (z ∈ R¹⁰⁰) │
└─────────────────────┬─────────────────────────────────────
──┘
│
▼
┌───────────────────────┐
│ 2-Layer LSTM Stack │
│ (256 hidden units) │
└───────────┬───────────┘
│
▼
┌───────────────────────┐
│ Reshape to 8×8×256 │
└───────────┬───────────┘
│
▼
┌────────────────────────────────┐
│ Transposed Conv Layers (×4) │
│ + Batch Norm + ReLU │
│ 8×8 → 16×16 → 32×32 → 64×64 │
│ → 128×128 │
└────────────────┬───────────────┘
│
▼
┌───────────────────────┐
│ Tanh Activation │
│ Output: 128×128×3 │
└───────────────────────┘
```
### Key Components
1. **Generator**: Noise → LSTM → Transposed Convolutions → Synthetic Image
2. **Critic**: Real/Fake Images → CNN → Wasserstein Distance Estimation
3. **Classifier**: Feature Extraction (CNN/EfficientNet) → LSTM → Sigmoid → Prediction
---
## 📊 Dataset
### Dataset Requirements
- **Minimum Size**: 1,000+ images (5,000+ recommended)
- **Format**: JPG, PNG
- **Resolution**: 256×256 pixels (will be resized automatically)
- **Labels**: Binary (infected/non_infected)
- **Annotations**: Clinical validation preferred
### Data Preprocessing
Our pipeline includes:
- ✅ Image resizing to 256×256
- ✅ Pixel normalization [0, 1]
- ✅ Data augmentation (rotation, flipping, zoom, contrast)
- ✅ Stratified train/val/test split (70:15:15)
### Obtaining Licensed Medical Datasets
⚠️ **Important**: This project requires clinically validated, ethically sourced medical imaging
data. Please ensure:
- Appropriate ethical approvals
- Patient consent and anonymization
- Compliance with HIPAA/GDPR regulations
- Institutional review board (IRB) approval
**Recommended Sources**:
- Hospital/clinical partnerships
- Licensed medical image repositories
---
## 🎓 Model Training
### Training Configuration
#### WGAN-GP + LSTM (Recommended)
```python
# Best performing configuration
{
'batch_size': 64,
'epochs': 200,
'lr_generator': 0.0001,
'lr_critic': 0.0002,
'lstm_layers': 2,
'lstm_hidden': 256,
'dropout': 0.3,
'gradient_penalty': 10
}
```
### Training Process
1. **Generator Training**:
- Input: Random noise vector (100-dim)
- LSTM processes latent sequences
- Transposed convolutions upsample to 128×128
- Critic evaluates Wasserstein distance
2. **Critic Training**:
- Enforces Lipschitz continuity via gradient penalty
- Provides stable training signals
- Prevents mode collapse
3. **Classifier Training**:
- Uses synthetic + real images
- Feature extraction → LSTM → Classification
- Binary cross-entropy loss
### Monitoring Training
```bash
# View training logs
tensorboard --logdir runs/
# Monitor metrics in real-time
python monitor.py --log_dir logs/
```
### Early Stopping
Training automatically stops when:
- FID score plateaus for 10 consecutive epochs
- Validation loss increases for 15 epochs
- Maximum epoch limit reached (200)
---
## 📈 Evaluation
### Generative Quality Metrics
| Metric | Description | Best Value | Our Result |
|--------|-------------|------------|------------|
| **Inception Score (IS)** | Quality & diversity | 9.0+ | **7.10** |
| **FID** | Distribution similarity | <10 | **28** |
| **PSNR** | Pixel-wise fidelity | >40 dB | **35 dB** |
| **SSIM** | Structural similarity | 1.0 | **0.85** |
### Classification Performance
| Metric | Value |
|--------|-------|
| **Accuracy** | 87.0% |
| **Precision** | 0.88 |
| **Recall** | 0.89 |
| **F1-Score** | 0.885 |
| **AUROC** | 0.93 |
| **TPR** | 0.87 |
| **FPR** | 0.13 |
