import torch
import cv2
import numpy as np
from torchvision import models, transforms
import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

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
print("ResNet models loaded successfully!")
resnet34.eval()
resnet50.eval()
resnet101.eval()

# Preprocessing for ResNet
resnet_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def extract_features_and_labels(folder_path, label):
    features, labels = [], []
    
    for filename in os.listdir(folder_path):
        if filename.endswith(('.jpg', '.png', '.jpeg')):
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)
            if image is None:
                continue
            
            # Convert to RGB and preprocess
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img_tensor = resnet_transform(image_rgb).unsqueeze(0)
            
            with torch.no_grad():
                softmax34 = torch.nn.functional.softmax(resnet34(img_tensor), dim=1).squeeze().numpy()
                softmax50 = torch.nn.functional.softmax(resnet50(img_tensor), dim=1).squeeze().numpy()
                softmax101 = torch.nn.functional.softmax(resnet101(img_tensor), dim=1).squeeze().numpy()
            
            # Concatenate softmax probabilities from all models
            feature_vector = np.concatenate([softmax34, softmax50, softmax101])
            features.append(feature_vector)
            labels.append(label)  # 0 for biodegradable, 1 for non-biodegradable
    
    return np.array(features), np.array(labels)

# Paths to save extracted features
feature_dir = "staking"
os.makedirs(feature_dir, exist_ok=True)

bio_feat_path = os.path.join(feature_dir, "bio_features.npy")
bio_label_path = os.path.join(feature_dir, "bio_labels.npy")
non_bio_feat_path = os.path.join(feature_dir, "non_bio_features.npy")
non_bio_label_path = os.path.join(feature_dir, "non_bio_labels.npy")

# Load or extract features
if os.path.exists(bio_feat_path) and os.path.exists(non_bio_feat_path):
    print("Loading saved features...")
    bio_features = np.load(bio_feat_path)
    bio_labels = np.load(bio_label_path)
    non_bio_features = np.load(non_bio_feat_path)
    non_bio_labels = np.load(non_bio_label_path)
else:
    print("Extracting features from images...")
    bio_features, bio_labels = extract_features_and_labels("staking dataset/biodegradable", label=0)
    non_bio_features, non_bio_labels = extract_features_and_labels("staking dataset/non_biodegradable", label=1)
    
    # Save extracted features
    np.save(bio_feat_path, bio_features)
    np.save(bio_label_path, bio_labels)
    np.save(non_bio_feat_path, non_bio_features)
    np.save(non_bio_label_path, non_bio_labels)
    print("Features saved in 'staking' folder for future runs.")

# Combine and shuffle dataset
X = np.vstack((bio_features, non_bio_features))
y = np.hstack((bio_labels, non_bio_labels))

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
tensor_X_train = torch.tensor(X_train, dtype=torch.float32)
tensor_y_train = torch.tensor(y_train, dtype=torch.long)
tensor_X_val = torch.tensor(X_val, dtype=torch.float32)
tensor_y_val = torch.tensor(y_val, dtype=torch.long)

# Create DataLoader
train_dataset = TensorDataset(tensor_X_train, tensor_y_train)
val_dataset = TensorDataset(tensor_X_val, tensor_y_val)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Define a simple neural network meta-learner
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

# Train the meta-learner
meta_learner = MetaLearner()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(meta_learner.parameters(), lr=0.001)

num_epochs = 30
for epoch in range(num_epochs):
    meta_learner.train()
    total_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = meta_learner(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader)}")

# Save the trained meta-learner
torch.save(meta_learner.state_dict(), "models/meta_learner.pt")
print("Meta-learner training complete and saved!")
