import numpy as np

# Print the predictions for the biodegradable images
predictions = np.load("predictions_testing_biodegradable.npy")
print(predictions)
count_0 = np.sum(predictions == 0)
count_1 = np.sum(predictions == 1)
accuracy = count_0 / (count_0 + count_1)
print(f"Number of 0s: {count_0}")
print(f"Number of 1s: {count_1}")
print(f"Accuracy: {accuracy}")

# Print the predictions for the non-biodegradable images
predictions = np.load("predictions_testing_non_biodegradable.npy")
print(predictions)
count_0 = np.sum(predictions == 0)
count_1 = np.sum(predictions == 1)
accuracy = count_1 / (count_0 + count_1)
print(f"Number of 0s: {count_0}")
print(f"Number of 1s: {count_1}")
print(f"Accuracy: {accuracy}")
