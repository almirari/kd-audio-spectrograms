# Digital Lung Stethoscope with CNN and Knowledge Distillation for Respiratory Disease Detection

> **Platform:** Raspberry Pi 4 (Edge Computing)  
> **Method:** Convolutional Neural Network (CNN) + Knowledge Distillation  

Respiratory illnesses such as Pneumonia, COPD (PPOK), and Bronchitis remain a major health problem, especially in remote regions where access to specialists and reliable internet is limited.

This project focuses on developing a **portable digital stethoscope** that can perform early screening of respiratory diseases **directly on the device**, without relying on cloud computing. To make the system efficient, we implemented **Knowledge Distillation**, allowing a larger neural network to guide a smaller model that runs smoothly on embedded hardware while preserving accuracy.

---

## Key Features

- **On-Device Inference:** All processing runs locally on Raspberry Pi — no internet required.  
- **Disease Classification:** Detects four classes of lung sounds: **Healthy, Asthma, COPD, and Pneumonia**.  
- **Model Compression:** Model size reduced to **≤ 5 MB** using knowledge distillation.  
- **Fast Inference:** Average processing time is about **183 ms per recording**.  
- **Portable Design:** Battery-powered and equipped with an OLED screen for direct feedback.

---

## System Architecture

### 1. Hardware Components
- **Processor:** Raspberry Pi 4 Model B  
- **Sensor:** Modified digital stethoscope (anterior/posterior chest placement)  
- **Display:** OLED screen (shows label + confidence score)  
- **Power:** Rechargeable 3.7V Li-ion battery  
- **Storage:** MicroSD or internal storage  

### 2. Software Pipeline

1. **Recording:** Captures 5–10 seconds of lung audio in `.wav` format.  
2. **Preprocessing:**  
   - Segmentation using energy thresholding  
   - Band-pass filtering (100–2000 Hz) to reduce noise  
   - Conversion to **Mel-spectrogram**  
3. **Inference:** Processed by the compressed **MobileNetV2 Student Model**.  
4. **Output:** Diagnosis and confidence score appear on the OLED display.

---

## Knowledge Distillation Approach

Knowledge Distillation was used to train a lighter model by transferring patterns learned from a larger, more expressive network.

| Feature | ResNet50 (Teacher) | MobileNetV2 (Student) |
|---|---|---|
| Role | Full-capacity training model | Deployed model |
| Training | Hard labels only | Hard labels + soft teacher outputs |
| Size | ~23.5M parameters | ~2.2M parameters |

MobileNetV2 was selected for deployment due to its efficiency in embedded environments, while ResNet50 served as a strong baseline during training.

---

## Results & Evaluation

Testing was conducted using the **ICBHI 2017** and **Mendeley Pulmonary Sound** datasets. Interestingly, the distilled Student model performed slightly better than the Teacher model on the validation set — likely due to the regularization effect of KD.

| Metric | ResNet50 | **MobileNetV2 (Final)** |
|---|---:|---:|
| Parameters | 23.5M | **2.2M** |
| Accuracy | 81.67% | **84.17%** |
| F1-Score | 0.8178 | **0.8436** |
| Sensitivity | 0.8167 | **0.8417** |
| Specificity | 0.9389 | **0.9472** |
| Precision | 0.8274 | **0.8495** |

### Runtime Efficiency
- **Inference Time:** ~183 ms  
- **Power Usage:** < 2.5W  

---

## Authors

- Almira Raisa Izzatina — Informatics Engineering, ITS, Surabaya
- Ziyan Nadia Putri — Medical Technology, ITS, Surabaya
- Giovrey Ernesto Putra — Medical Technology, ITS, Surabaya
- Putri Alief Siswanto — Medical Technology, ITS, Surabaya
