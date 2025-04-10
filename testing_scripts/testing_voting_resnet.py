import torch
import cv2
import numpy as np
from ultralytics import YOLO
from torchvision import models, transforms
from PIL import Image
import os

# Load YOLO models
model_v8s = YOLO("models/yolov8s_best.pt")
model_v8m = YOLO("models/yolov8m_best.pt")
model_v9s = YOLO("models/yolov9s_best.pt")
print('YOLO models loaded successfully!')

# Load ResNet models
resnet34 = models.resnet34()
resnet34.fc = torch.nn.Linear(in_features=512, out_features=2)
resnet50 = models.resnet50()
resnet50.fc = torch.nn.Linear(in_features=2048, out_features=2)
resnet101 = models.resnet101()
resnet101.fc = torch.nn.Linear(in_features=2048, out_features=2)

# Load fine-tuned weights
resnet34.load_state_dict(torch.load("models/resnet34_biodegradable_best.pt", map_location=torch.device('cpu')))
resnet50.load_state_dict(torch.load("models/resnet50_biodegradable_best.pt", map_location=torch.device('cpu')))
resnet101.load_state_dict(torch.load("models/resnet101_biodegradable_best.pt", map_location=torch.device('cpu')))

resnet34.eval()
resnet50.eval()
resnet101.eval()
print('ResNet models loaded successfully!')

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
    
    all_boxes = np.vstack(all_boxes)  # Merge detections
    scores = all_boxes[:, 4]  # Confidence scores
    boxes = all_boxes[:, :4]  # Bounding box coordinates
    class_ids = all_boxes[:, 5].astype(int)  # Class labels

    # Apply Non-Maximum Suppression (NMS)
    indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), score_threshold=0.5, nms_threshold=iou_threshold)
    
    final_boxes = []
    if len(indices) > 0:
        for i in indices.flatten():
            final_boxes.append((*boxes[i], class_ids[i]))
    
    return final_boxes

def classify_with_resnet_voting(image, box):
    x1, y1, x2, y2, class_id = map(int, box)
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    img_tensor = resnet_transform(crop).unsqueeze(0)
    
    with torch.no_grad():
        pred34 = torch.argmax(resnet34(img_tensor)).item()
        pred50 = torch.argmax(resnet50(img_tensor)).item()
        pred101 = torch.argmax(resnet101(img_tensor)).item()
        
        # Majority voting mechanism
        votes = [pred34, pred50, pred101]
        final_class_id = max(set(votes), key=votes.count)
        final_class = 0 if final_class_id == 0 else 1  # 0: Biodegradable, 1: Non-Biodegradable
    
    return final_class

def process_images_from_folder(folder_path):
    predictions = []
    
    for filename in os.listdir(folder_path):
        if filename.endswith(('.jpg', '.png', '.jpeg')):
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)
            
            if image is None:
                print(f"Skipping {filename}, unable to read.")
                continue
            
            boxes = ensemble_yolo_predictions(image)
            if not boxes:
                print(f"No objects detected in {filename}.")
                continue
            
            for box in boxes:
                prediction = classify_with_resnet_voting(image, box)
                if prediction is not None:
                    predictions.append(prediction)
    
    np.save("predictions_non-biodegradable.npy", np.array(predictions))
    print("Predictions saved successfully!")

# Folder containing images
test_folder = "dataset/test/non_biodegradable"
process_images_from_folder(test_folder)
