import os
import random
import pandas as pd

# Folder containing the original CSV files
input_folder = r"C:\Users\tunja\OneDrive\Работен плот\Experiment-3\Data"

# Output folder
output_folder = r"C:\Users\tunja\OneDrive\Работен плот\Experiment-3\Experiment3"

train_folder = os.path.join(output_folder, "train")
validation_folder = os.path.join(output_folder, "validation")

os.makedirs(train_folder, exist_ok=True)
os.makedirs(validation_folder, exist_ok=True)

# Group files by class
class_files = {}

for file in os.listdir(input_folder):
    if file.endswith(".csv"):
        label = file.split("_")[0].lower()

        if label != "lying":
            label = label.replace("ing", "")

        if label == "sitt":
            label = label.replace("tt", "t")

        if label not in class_files:
            class_files[label] = []

        class_files[label].append(file)

# Split each class
random.seed(42)

for label, files in class_files.items():

    random.shuffle(files)

    split_index = int(len(files) * 0.8)

    train_files = files[:split_index]
    validation_files = files[split_index:]

    train_class_folder = os.path.join(train_folder, label)
    validation_class_folder = os.path.join(validation_folder, label)

    os.makedirs(train_class_folder, exist_ok=True)
    os.makedirs(validation_class_folder, exist_ok=True)

    # Save training files
    for file in train_files:

        csv_path = os.path.join(input_folder, file)
        df = pd.read_csv(csv_path).copy()

        df["Label"] = label

        output_path = os.path.join(train_class_folder, file)
        df.to_csv(output_path, index=False)

        print(f"Train: {output_path}")

    # Save validation files
    for file in validation_files:

        csv_path = os.path.join(input_folder, file)
        df = pd.read_csv(csv_path).copy()

        df["Label"] = label

        output_path = os.path.join(validation_class_folder, file)
        df.to_csv(output_path, index=False)

        print(f"Validation: {output_path}")