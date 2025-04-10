import numpy as np

# Load predictions from individual ResNet models
predictions_resnet34_bio = np.load("predictions_resnet34_bio.npy")
predictions_resnet50_bio = np.load("predictions_resnet50_bio.npy")
predictions_resnet101_bio = np.load("predictions_resnet101_bio.npy")

predictions_resnet34_non_bio = np.load("predictions_resnet34_non_bio.npy")
predictions_resnet50_non_bio = np.load("predictions_resnet50_non_bio.npy")
predictions_resnet101_non_bio = np.load("predictions_resnet101_non_bio.npy")

# Function to calculate and print prediction stats
def evaluate_predictions_bio(predictions, model_name):
    count_0 = np.sum(predictions == 0)
    count_1 = np.sum(predictions == 1)
    accuracy = count_0 / (count_0 + count_1)

    print(f"Results for {model_name}:")
    print(f"Number of 0s: {count_0}")
    print(f"Number of 1s: {count_1}")
    print(f"Accuracy: {accuracy:.2%}")
    print("-" * 30)
def evaluate_predictions_non_bio(predictions, model_name):
    count_0 = np.sum(predictions == 0)
    count_1 = np.sum(predictions == 1)
    accuracy = count_1 / (count_0 + count_1)

    print(f"Results for {model_name}:")
    print(f"Number of 0s: {count_0}")
    print(f"Number of 1s: {count_1}")
    print(f"Accuracy: {accuracy:.2%}")
    print("-" * 30)
# Evaluate each model
evaluate_predictions_bio(predictions_resnet34_bio, "ResNet34")
evaluate_predictions_non_bio(predictions_resnet34_non_bio, "ResNet34")

evaluate_predictions_bio(predictions_resnet50_bio, "ResNet50")
evaluate_predictions_non_bio(predictions_resnet50_non_bio, "ResNet50")

evaluate_predictions_bio(predictions_resnet101_bio, "ResNet101")
evaluate_predictions_non_bio(predictions_resnet101_non_bio, "ResNet101")