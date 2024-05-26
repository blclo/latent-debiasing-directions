"""
After running the script ´pipeline_sdxl_save_latents.py´ you must have ended up with a json dictionary
containing the image path, all its latents, and their respective label (e.g. woman, black_skin, etc.).
"""

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import os
import torch
import pickle
import json

path_dict = "path_to_your_saved_dictionary.json"
class_1 = "class1"
class_2 = "class2"

file = open(f"{path_dict}")
json_file = json.load(file)

# obtain latent directions of latents at different latent steps
for idx_latent in range(10):
    # Get latent codes from directory containing them and iterate through them
    latent_codes = []
    labels = []

    for i in range(len(json_file)):
        image_class = json_file[i]["class"]
        if image_class == class_1:
            class1_latent_path = json_file[i]["latents"][idx_latent]
            with open(class1_latent_path, 'rb') as class1_latent:
                class1_latent = pickle.load(class1_latent)
            class_latent = class1_latent.cpu().numpy()
            latent_code = np.squeeze(class_latent)  # Remove the extra dimension
            latent_code = latent_code.flatten()  # Flatten the latent code
            latent_codes.append(latent_code)
            labels.append(1)
        elif image_class == class_2:
            class2_latent_path = json_file[i]["latents"][idx_latent]
            with open(class2_latent_path, 'rb') as class2_latent:
                class2_latent = pickle.load(class2_latent)
            class_latent = class2_latent.cpu().numpy()
            latent_code = np.squeeze(class_latent)  # Remove the extra dimension
            latent_code = latent_code.flatten()  # Flatten the latent code
            latent_codes.append(latent_code)
            labels.append(0)
        else:
            continue

    # Convert the latent codes and labels to numpy arrays
    latent_codes = np.array(latent_codes)
    print("Initial latent_codes shape: ", latent_codes.shape)
    labels = np.array(labels)

    # Step 2: Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(latent_codes, labels, test_size=0.4, random_state=42)

    # Convert the training data to torch tensors
    X_train = torch.from_numpy(X_train).float()
    print("Initial shape: ", X_train.shape)
    y_train = torch.from_numpy(y_train)

    # Reshape X_train to the desired shape (1, 18, 512)
    X_train = X_train.view(X_train.shape[0], *X_train.shape[1:])

    # Step 4: Train the SVM model
    svm_classifier = svm.SVC(kernel='linear')  # Create an SVM classifier with linear kernel
    svm_classifier.fit(X_train, y_train)  # Fit the classifier to the training data
    print(X_train.shape)
    print(y_train.shape)

    # Step 5: Evaluate the model
    X_test = torch.from_numpy(X_test).float()
    X_test = X_test.view(X_test.shape[0], *X_test.shape[1:])
    y_pred = svm_classifier.predict(X_test)  # Predict labels for the test data
    accuracy = accuracy_score(y_test, y_pred)  # Calculate accuracy
    print("Accuracy:", accuracy)


    # Get the support vectors and corresponding coefficients
    support_vectors = svm_classifier.support_vectors_
    coefficients = svm_classifier.coef_[0]

    print(coefficients.shape)
    # Step 6: Normalize the weight vector to obtain the latent direction
    latent_direction = coefficients / np.linalg.norm(coefficients)

    class1_direction = latent_direction
    # To get the class 2 direction, negate the latent direction
    class2_direction = -latent_direction

    print(latent_direction.shape)
    # Specify the original shape to unflatten the array
    original_shape = (1, 4, 128, 128)

    # Unflatten the array
    unflattened_array_class1 = class1_direction.reshape(original_shape)
    print(unflattened_array_class1.shape)

    # Save the array as a file
    np.save(f'gendered_latents/latent_direction_{class_1}_{idx_latent}.npy', unflattened_array_class1)

    # Unflatten the array
    unflattened_array_class2 = class2_direction.reshape(original_shape)
    print(unflattened_array_class2.shape)

    # Save the array as a file
    np.save(f'gendered_latents/latent_direction_{class_2}_{idx_latent}.npy', unflattened_array_class2)
    print("Saved latent directions.")