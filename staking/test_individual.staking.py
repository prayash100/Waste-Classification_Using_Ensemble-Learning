import torch
import cv2
import numpy as np
from ultralytics import YOLO
from torchvision import models, transforms
import torch.nn as nn
import matplotlib.pyplot as plt

# Load YOLO models
print("Loading YOLO models...")
model_v8s = YOLO("models/yolov8s_best.pt")
model_v8m = YOLO("models/yolov8m_best.pt")
model_v9s = YOLO("models/yolov9s_best.pt")
print("YOLO models loaded successfully!")

# Load ResNet models
print("Loading ResNet models...")
resnet34 = models.resnet34()
resnet34.fc = torch.nn.Linear(in_features=512, out_features=2)
resnet50 = models.resnet50()
resnet50.fc = torch.nn.Linear(in_features=2048, out_features=2)
resnet101 = models.resnet101()
resnet101.fc = torch.nn.Linear(in_features=2048, out_features=2)

resnet34.load_state_dict(torch.load("models/resnet34_biodegradable_best.pt", map_location=torch.device('cpu')))
resnet50.load_state_dict(torch.load("models/resnet50_biodegradable_best.pt", map_location=torch.device('cpu')))
resnet101.load_state_dict(torch.load("models/resnet101_biodegradable_best.pt", map_location=torch.device('cpu')))

resnet34.eval()
resnet50.eval()
resnet101.eval()
print("ResNet models loaded successfully!")

# Load trained meta-learner
print("Loading Meta-Learner model...")
class MetaLearner(nn.Module):
    def __init__(self):
        super(MetaLearner, self).__init__()
        self.fc1 = nn.Linear(6, 4)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(4, 2)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.softmax(self.fc2(x))
        return x

meta_learner = MetaLearner()
meta_learner.load_state_dict(torch.load("models/meta_learner.pt", map_location=torch.device('cpu')))
meta_learner.eval()
print("Meta-Learner model loaded successfully!")
print("************************************************************")
# Preprocessing for ResNet
resnet_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def ensemble_yolo_predictions(image, iou_threshold=0.5):
    results = [model.predict(image)[0] for model in [model_v8s, model_v8m, model_v9s]]
    all_boxes = []
    
    for result in results:
        if result.boxes is not None:
            all_boxes.append(result.boxes.data.cpu().numpy())
    
    if not all_boxes:
        return []
    
    all_boxes = np.vstack(all_boxes)
    scores = all_boxes[:, 4]
    boxes = all_boxes[:, :4]
    class_ids = all_boxes[:, 5].astype(int)

    indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), score_threshold=0.5, nms_threshold=iou_threshold)
    
    final_boxes = []
    if len(indices) > 0:
        for i in indices.flatten():
            final_boxes.append((*boxes[i], class_ids[i]))
    
    return final_boxes

def extract_features(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_tensor = resnet_transform(image_rgb).unsqueeze(0)
    
    with torch.no_grad():
        softmax34 = torch.nn.functional.softmax(resnet34(img_tensor), dim=1).squeeze().numpy()
        softmax50 = torch.nn.functional.softmax(resnet50(img_tensor), dim=1).squeeze().numpy()
        softmax101 = torch.nn.functional.softmax(resnet101(img_tensor), dim=1).squeeze().numpy()
    
    return np.concatenate([softmax34, softmax50, softmax101])

def process_single_image(image_path):
    print(f"Processing image: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Unable to read the image.")
        return
    
    boxes = ensemble_yolo_predictions(image)
    if not boxes:
        print("No objects detected, passing full image to ResNet.")
        features = extract_features(image)
    else:
        for box in boxes:
            x1, y1, x2, y2, class_id = map(int, box)
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            features = extract_features(crop)
            
            input_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                output = meta_learner(input_tensor)
                prediction = torch.argmax(output).item()
            
            label = "Non-Biodegradable" if prediction == 1 else "Biodegradable"
            color = (0, 0, 255) if prediction == 1 else (0, 255, 0)  # Red for non-biodegradable, Green for biodegradable
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            print(f"Final Prediction: {label}")
    
    plt.figure(figsize=(8, 6))
    plt.axis("off")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()

# Example usage
image_path = "staking dataset/non_biodegradable/glass-113-_jpg.rf.28e303db6570a220d07a3057ec807feb.jpg"
process_single_image(image_path)
