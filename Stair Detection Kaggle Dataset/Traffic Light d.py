import kagglehub

import json
from PIL import Image
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn

#from models.FasterRCNN import Backbone, TrafficLightColourDetector, CustomBackbone, DetectionHead
from testing_functions import forward_pass, compute_losses,  generate_anchors, plot_dataset_item

from dsets import  TrafficLightDataset, load_dataset

#from training import TrainingApp

from ultralytics import YOLO
#yolo=YOLO("yolov8n.pt")

#print(yolo.model)
#backbone=yolo.model[0]
#print(backbone.shape)

# Download latest version
path = kagglehub.dataset_download("wjybuqi/traffic-light-detection-dataset")
print(f"PyTorch version: {torch.__version__}")
print("Path to dataset files:", path)



base_path = r"c:\users\alex\.cache\kagglehub\datasets\wjybuqi\traffic-light-detection-dataset\versions\4"
print("Folders in dataset directory:")
print(os.listdir(base_path))
print("Folders in training directory")
print(os.listdir(os.path.join(base_path, "train_dataset")))
print("Folders in testing directory")
print(os.listdir(os.path.join(base_path, "test_dataset")))
json_path = os.path.join(base_path, "train_dataset", "train.json")

if os.path.exists(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print("Successfully loaded JSON file.")
    print("Keys in JSON data:", data.keys())

else:
    print(f"JSON file not found at {json_path}")

first_ann = data['annotations'][0] 
print("First annotation:", first_ann)



submit_example_path=os.path.join(base_path, "'submit_example.json")
train_dataset=os.path.join(base_path, "train_dataset")
test_dataset=os.path.join(base_path, "test_dataset")
#print(os.listdir(train_dataset)) 
#print(os.listdir(test_dataset))

input_json_train=os.path.join(train_dataset, "train.json")
readable_training_json=r"C:\Users\Alex\source\repos\Stair Detection Kaggle Dataset\readable_training_json.json"
input_json_test=os.path.join(test_dataset, "test_images")
readable_test_json=r"C:\Users\Alex\source\repos\Stair Detection Kaggle Dataset\readable_test_json.json"

with open(input_json_train, "r", encoding="utf-8") as f:
    data_train = json.load(f)

with open(readable_training_json, "w", encoding="utf-8") as f:
    json.dump(data_train, f, indent=4)




data_train = load_dataset()

print(f"Loaded {len(data_train)} training samples.")
print(f"Number of images: {len(set(item['filename'] for item in data_train))}")

base_path = r"c:\users\alex\.cache\kagglehub\datasets\wjybuqi\traffic-light-detection-dataset\versions\4"
dataset = TrafficLightDataset(dataset_path=base_path, mode='train')

image, target, image_name = dataset[0]
print("Image name:", image_name)
print("Image shape:", image.shape)
print("Number of boxes:", target['boxes'].shape[0])
print("Boxes:", target['boxes'])
print("Labels shape:", target['labels'].shape)

for idx in range(5):
    plot_dataset_item(dataset, idx)

print(f"Number of classes:{len(dataset.label_map)}")

test_images=load_dataset(mode="test")

print(f"Loaded {len(test_images)} test samples.")

test_dataset = TrafficLightDataset(dataset_path=base_path, mode="test")

print(f"Number of test samples: {len(test_dataset)}")





















