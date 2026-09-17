# 🌱 CropGuard AI

## AI-Powered Crop Disease Detection and Environmental Risk Screening

CropGuard AI is an explainable computer vision system designed to identify tomato leaf diseases from images and provide an additional environmental risk assessment based on temperature, humidity, and recent rainfall.

The system combines a fine-tuned **ResNet18 image classification model**, confidence analysis and calibration, **Grad-CAM explainability**, a curated disease knowledge base, and a rule-based environmental risk engine into an interactive **Streamlit application**.

> **Important:** CropGuard AI is an AI-based screening and decision-support prototype. It is not a definitive agricultural diagnosis. Real-world treatment decisions should be verified using appropriate local agricultural guidance.

---

## 📌 Project Overview

Plant diseases can significantly affect crop productivity. Early identification of visible symptoms can help farmers and agricultural workers monitor affected plants and take appropriate action.

CropGuard AI addresses this problem through two complementary components:

1. **Image-based disease screening**
   - A tomato leaf image is analyzed by a fine-tuned ResNet18 model.
   - The model predicts the most likely disease class.
   - Top-3 predictions and model scores are displayed.

2. **Environmental risk assessment**
   - Temperature, relative humidity, and recent rainfall are provided by the user.
   - A disease-specific rule engine evaluates whether the supplied conditions are favorable for disease development or pest activity.

The system also provides **Grad-CAM visual explanations** showing which regions of the image contributed strongly to the model's prediction.

---

## 🎯 Objectives

The main objectives of CropGuard AI are:

- Detect tomato leaf diseases using computer vision.
- Build and evaluate a transfer-learning based image classifier.
- Analyze model errors and confusion between disease classes.
- Evaluate model robustness under altered image conditions.
- Calibrate model confidence scores.
- Provide interpretable predictions using Grad-CAM.
- Combine image-based screening with environmental conditions.
- Present the complete workflow through an interactive web application.
- Maintain a reproducible and well-structured machine learning codebase.

---

## ✨ Key Features

### 🔬 AI Disease Detection

Uses a fine-tuned **ResNet18** convolutional neural network to classify tomato leaf images into 10 classes.

### 📊 Top-3 Predictions

Instead of showing only one prediction, CropGuard AI displays the model's three highest-scoring classes.

### 🎯 Confidence Screening

The system applies a configurable screening threshold to distinguish predictions that meet the current confidence criterion from lower-confidence predictions.

### 📐 Confidence Calibration

Temperature scaling is used to improve the alignment between model confidence and observed prediction correctness.

### 🧠 Grad-CAM Explainability

Grad-CAM highlights image regions that contributed strongly to the model's prediction.

This makes the model's decision more interpretable instead of treating the classifier as a complete black box.

### 🌦️ Environmental Risk Engine

The system evaluates disease-specific environmental conditions using:

- Temperature
- Relative humidity
- Recent rainfall

The engine produces:

- Risk level
- Risk score
- Contributing environmental factors
- Monitoring recommendations

### 📚 Disease Knowledge Base

A structured disease information file provides:

- Disease type
- Description
- Common symptoms
- Management guidance
- Prevention guidance

### 🌐 Interactive Streamlit Application

The complete pipeline is exposed through a browser-based interface where users can:

1. Upload a tomato leaf image.
2. View the AI prediction.
3. Inspect the Top-3 predictions.
4. Read disease information.
5. Enter environmental conditions.
6. View environmental risk.
7. Inspect Grad-CAM model attention.
8. View the combined screening assessment.

---

# 🏗️ System Architecture

```text
                    ┌──────────────────────┐
                    │   Tomato Leaf Image  │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Image Preprocessing  │
                    │ Resize + Normalize   │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Fine-Tuned ResNet18  │
                    │   Vision Model       │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Disease Prediction   │
                    │ + Confidence Score   │
                    └───────┬───────┬──────┘
                            │       │
                ┌───────────┘       └─────────────┐
                ▼                                 ▼
       ┌─────────────────┐              ┌──────────────────┐
       │ Disease         │              │ Grad-CAM         │
       │ Knowledge Base  │              │ Explanation      │
       └────────┬────────┘              └──────────────────┘
                │
                ▼
       ┌──────────────────────┐
       │ Environmental Risk   │
       │ Engine               │
       │                      │
       │ Temperature          │
       │ Humidity             │
       │ Rainfall             │
       └──────────┬───────────┘
                  │
                  ▼
       ┌──────────────────────┐
       │ Combined Assessment  │
       └──────────┬───────────┘
                  │
                  ▼
       ┌──────────────────────┐
       │ Streamlit Web App    │
       └──────────────────────┘