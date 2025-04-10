# 🗑️ Waste Classification Using Ensemble Learning

This project uses ensemble learning techniques to intelligently classify waste as **biodegradable** or **non-biodegradable**. It integrates **YOLO-based object detection**, **ResNet-based image classification**, and a custom **meta-learner neural network** to enhance prediction accuracy.

---

## 🚀 Features

- 🔍 YOLOv8/v9-based object detection:
  - YOLOv8s, YOLOv8m, YOLOv9s
- 🧠 ResNet-based classifiers:
  - ResNet34, ResNet50, ResNet101
- 🧬 Meta-learner neural network that combines predictions from ResNet models
- 🖼️ Automatic cropping of detected objects and per-object classification
- 📦 Real-time visual output with bounding boxes and labels
- ✅ Ensemble Non-Maximum Suppression (NMS) for robust object filtering

---

## 🧠 Model Architecture

### 🔹 Object Detection
- Runs YOLOv8s, YOLOv8m, and YOLOv9s on the input image
- Applies NMS to remove overlapping detections

### 🔹 Feature Extraction
- Detected regions are cropped from the image
- Crops are passed through:
  - ResNet34
  - ResNet50
  - ResNet101
- Each model outputs softmax probabilities for:
  - Biodegradable
  - Non-biodegradable

### 🔹 Meta-Learner
- Receives a 6-length feature vector (concatenated softmax outputs)
- Processes through a shallow neural network
- Produces the final classification decision

---

## 📷 Example Output

Bounding boxes with color-coded labels:
- 🟢 Biodegradable
- 🔴 Non-Biodegradable

---

Feel free to contribute or fork this repo to enhance the model, expand the dataset, or explore other ensemble strategies!

