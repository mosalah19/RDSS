# Retinal Diseases Diagnosis Support System  

**Graduation Project**  
**Contributors**: Mohamed Salah, Asmaa  
📅 **Completion Date**: April 29, 2024  

---

## 🌟 Project Overview  
This project aims to provide an AI-powered diagnostic system for **retinal disease detection**. Using a **smartphone** with a **20D lens attachment**, the system captures high-resolution retinal images, processes them using AI models, and offers actionable insights for ophthalmologists.

---

## 🛠️ Features  
- **Disease Classification**: Cataract, Glaucoma, Diabetic Retinopathy (DR) stages.  
- **Blood Vessel Segmentation**: Semantic segmentation for vascular analysis.  
- **Mobile App Integration**: Available on Android and iOS for real-time use.

---

## 🔍 Problem Definition  

**Input**: Retinal fundus images captured with a smartphone.  
**Output**:  
- Disease classification (e.g., NoDR, Mild, Moderate, Severe, Proliferative).  
- Segmentation of blood vessels for further analysis.

---

## 🚀 Algorithms and Models  

### 1. **Preprocessing**  
- Image normalization and resizing.  
- Grayscale conversion with Gaussian blurring.

### 2. **Classification Models**  
- **Binary Classification**: VGG-16, MobileNet V3 Small, MobileNet V3 Large.  
- **Multi-Class Classification**: EfficientNet B5 for DR stages.  

### 3. **Semantic Segmentation**  
- **Model**: U-Net with ResNet-50 encoder.  

---

## 📊 Datasets  

| **Dataset**                       | **Use Case**                   | **Details**                         |
|------------------------------------|---------------------------------|-------------------------------------|
| ODIR                              | Binary Classification (Cataract) | 5000 patient cases.                |
| EyePACS-AIROGS-light-V2           | Glaucoma Detection             | 4000 train, 770 test/validation.   |
| APTOS 2019 Blindness Detection    | Multi-class DR Stages          | 4662 labeled images.               |
| Retina Blood Vessel Segmentation  | Vascular Abnormalities         | 100 images (80 train, 20 test).    |

---

## 📈 Results  

| **Task**                         | **Model**               | **Accuracy** | **Other Metrics**                  |
|-----------------------------------|-------------------------|--------------|-------------------------------------|
| Cataract Detection               | VGG-16                 | 98.58%       |                                     |
| Glaucoma Detection               | MobileNet V3 Small     | 88.0%        |                                     |
| Binary DR Classification         | MobileNet V3 Large     | 92.0%        |                                     |
| Multi-Class DR Classification    | EfficientNet B5        | 76.0%        | Balanced Dataset Improvement: +3%. |
| Blood Vessel Segmentation        | U-Net (ResNet-50)      | 96.5%        | IoU: 0.664, Precision: 0.805.      |

---
