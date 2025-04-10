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

def classify_with_resnet(image, box, model):
    x1, y1, x2, y2, class_id = map(int, box)
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    img_tensor = resnet_transform(crop).unsqueeze(0)
    
    with torch.no_grad():
        prediction = torch.argmax(model(img_tensor)).item()
        final_class = 0 if prediction == 0 else 1
    
    return final_class

def process_images_from_folder(folder_path):
    predictions_34 = []
    predictions_50 = []
    predictions_101 = []
    
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
                pred34 = classify_with_resnet(image, box, resnet34)
                pred50 = classify_with_resnet(image, box, resnet50)
                pred101 = classify_with_resnet(image, box, resnet101)
                
                if pred34 is not None:
                    predictions_34.append(pred34)
                if pred50 is not None:
                    predictions_50.append(pred50)
                if pred101 is not None:
                    predictions_101.append(pred101)
    
    np.save("predictions_resnet34_non_bio.npy", np.array(predictions_34))
    np.save("predictions_resnet50_non_bio.npy", np.array(predictions_50))
    np.save("predictions_resnet101_non_bio.npy", np.array(predictions_101))
    print("Predictions for individual ResNet models saved successfully!")

# Folder containing images
test_folder = "testing_dataset/non_biodegradable"
process_images_from_folder(test_folder)
