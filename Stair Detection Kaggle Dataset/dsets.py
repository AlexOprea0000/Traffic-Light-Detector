

from email.mime import image
import os
import json
from re import L
import torch
import imageio
from torch.utils.data import Dataset

import cv2 as cv


import random

from PIL import Image


import functools

import ast

base_path = r"c:\users\alex\.cache\kagglehub\datasets\wjybuqi\traffic-light-detection-dataset\versions\4"
train_dataset=os.path.join(base_path, "train_dataset", "train.json")

test_dataset=os.path.join(base_path, "test_dataset")

size=300






scale_factor = size / 1024




# @functools.lru_cache(1)
# def load_image_cached(image_path,transform=None ,requireOnDisk_bool=True):

#     if not(os.path.exists(image_path)):
#         raise FileNotFoundError(f"Image file not found at {image_path}")
        
#     #base_name = os.path.basename(image_path)
#     try:
#        img=cv.imread(image_path)
#        img=cv.cvtColor(img, cv.COLOR_BGR2RGB).astype('float32')/255.0
#        if transform:
#            img = transform(img)
#            return img

#        return img
       
#     except Exception as e:
#         print(f"Error loading image {image_path}: {e}")
#         return None

    
   

    

# @functools.lru_cache(1)
# def get_images_coordonates(requireOnDisk_bool=True, dataset_path=None, mode=None):
    
#     if mode=="train":
#         json_path = os.path.join(dataset_path, "train_dataset", "train.json")
#         img_folder = "train_dataset"
#     elif mode=="test":
#         json_path = os.path.join(dataset_path, "submit_example.json") 
#         img_folder = "test_dataset"
#     else:
#         print("Invalid mode. Please choose 'train' or 'test'.")
   
#     if not os.path.exists(json_path):
#         raise FileNotFoundError(f"JSON file not found at {json_path}")

#     with open(json_path, "r") as f:
#         data = json.load(f)

#     data_list={}
#     for annotation in data['annotations']:
#         img_rel_path = annotation['filename'].replace('\\', os.sep)
#         full_img_path = os.path.join(dataset_path, img_folder, img_rel_path)

        
#         # x_min = annotation['bndbox']['xmin']
#         # y_min = annotation['bndbox']['ymin']
#         # x_max = annotation['bndbox']['xmax']
#         # y_max = annotation['bndbox']['ymax']
        
#         current_image_data={
#             'filename': img_rel_path,
#             'objects': []}

#         if  annotation.get('inbox'):
#             for inbox in annotation['inbox']:
                
                

#                 current_image_data['objects'].append({
#                     'bbox': inbox['bndbox'],
#                     'color': inbox['color']
#                 })
#         if current_image_data['objects']:
#             data_list.append(current_image_data)

      
#     return data_list
      
        
        

@functools.lru_cache(maxsize=None)        
def load_dataset(requireOnDisk_bool=True, mode='train'):
    data_list=[]
    if mode == "train":
        json_path = os.path.join(base_path, "train_dataset", "train.json")
   
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"JSON file not found at {json_path}")

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        
        for annotation in data['annotations']:
            img_rel_path = annotation['filename'].replace('\\', os.sep)
        
            current_image_data={
               'filename': img_rel_path,
                'boxes': [],
                 'color': []
                }
            if  annotation.get('inbox'):
                for inbox in annotation['inbox']:
                
                    current_image_data['boxes'].append(inbox['bndbox'])
                    current_image_data['color'].append(inbox['color'])
            else:
                current_image_data['boxes'].append(annotation['bndbox'])
                current_image_data['color'].append("background")
                
            if current_image_data['boxes']:
                data_list.append(current_image_data)
    elif mode == "test":
        image_rel_path=os.path.join(base_path, "test_dataset", "test_images").replace('\\', os.sep)
        for img_name in os.listdir(image_rel_path):
            if img_name.endswith(('.jpg', '.jpeg', '.png')):
                data_list.append({'filename': img_name})
    
    return data_list   
    



@functools.lru_cache(maxsize=None)        
def load_dataset_mode2(requireOnDisk_bool=True, mode='train'):
    data_list = []
   
    images_map = {} 

    if mode == "train":
        json_path = os.path.join(base_path, "train_dataset", "train.json")
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for annotation in data['annotations']:
            fname = annotation['filename'].replace('\\', os.sep)
            
            
            if fname not in images_map:
                images_map[fname] = {'filename': fname, 'boxes': [], 'color': []}
            
            
            if annotation.get('inbox'):
                for inbox in annotation['inbox']:
                   
                    if str(inbox['color']) != "-1" and inbox['color'] != "background":
                        images_map[fname]['boxes'].append(inbox['bndbox'])
                        images_map[fname]['color'].append(inbox['color'])
            elif annotation.get('bndbox'):
                
                color = annotation.get('color', 'background')
                if color != "background" and str(color) != "-1":
                    images_map[fname]['boxes'].append(annotation['bndbox'])
                    images_map[fname]['color'].append(color)

       
        for fname in images_map:
            if images_map[fname]['boxes']:
                data_list.append(images_map[fname])

    elif mode == "test":
        
        path = os.path.join(base_path, "test_dataset", "test_images")
        for img_name in os.listdir(path):
            if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                data_list.append({'filename': img_name})
    
    return data_list



   
    
   
    
   







class TrafficLightDataset(Dataset):
    def __init__(self, dataset_path,resize_factor=0.25, mode="train", transform=None):
        self.dataset_path = dataset_path
        self.mode=mode
        self.transform = transform
        if mode == "train" or mode == 'val':
           self.data_list=load_dataset_mode2()
        elif mode == "test":
           self.data_list=load_dataset_mode2(mode="test")
        self.image_names=sorted(list(set(item['filename'] for item in self.data_list)))
        self.resize_factor=resize_factor
        self.annotation_dict={item['filename']: item for item in self.data_list}
       
        self.label_map={
            "background": 0,
            "red": 1,
            "yellow": 2,
            "green": 3
            }
      
    

    def __len__(self):
        return len(self.image_names)




    def __getitem__(self, idx):

        image_name= self.image_names[idx]
        
        if self.mode == "train" or self.mode == "val":
            image_path = os.path.join(self.dataset_path, "train_dataset", image_name)
        elif self.mode == "test":
            image_path = os.path.join(self.dataset_path, "test_dataset", "test_images", image_name)

        image=cv.imread(image_path, cv.IMREAD_COLOR)

        

        image=cv.cvtColor(image, cv.COLOR_BGR2RGB).astype('float32')/255.0

        if self.mode =="train" or self.mode =="val":
            annotation= self.annotation_dict[image_name]
            bboxes = annotation['boxes']
            bboxes = torch.tensor([[b['xmin'], b['ymin'], b['xmax'], b['ymax']] for b in bboxes], dtype=torch.float32)
           
            colors = annotation['color']
            labels = [self.label_map[color] for color in colors]

            height, width, _ = image.shape
            new_height, new_width = int(height * self.resize_factor), int(width * self.resize_factor)
            image = cv.resize(image, (new_width, new_height))
            bboxes *= self.resize_factor

            image = torch.from_numpy(image).permute(2, 0, 1) 
            

            if self.mode == "train":
                 if random.random() > 0.5:
                    image = torch.flip(image, dims=[2]) 
                    _, _, new_width = image.shape
                    old_xmin = bboxes[:, 0].clone()
                    old_xmax = bboxes[:, 2].clone()
                    bboxes[:, 0] = new_width - old_xmax
                    bboxes[:, 2] = new_width - old_xmin

            bboxes[:, [0, 2]] = bboxes[:, [0, 2]].clamp(min=0, max=new_width)
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]].clamp(min=0, max=new_height)
            
            area= (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
           

            target= {
                'boxes': bboxes,
                'labels': torch.tensor(labels, dtype=torch.int64),
                'area': torch.as_tensor(area, dtype=torch.float32),
                'iscrowd': torch.zeros((len(labels),), dtype=torch.int64),
                'image_id': torch.tensor([idx], dtype=torch.int64)
            }

            if self.transform:
                image = self.transform(image)
            
                
            
            return image, target, image_name


        


        elif self.mode == "test":

            height, width, _ = image.shape
            new_height, new_width = int(height * self.resize_factor), int(width * self.resize_factor)
            image = cv.resize(image, (new_width, new_height))
            image= torch.from_numpy(image).permute(2, 0, 1)
            if self.transform:
                image = self.transform(image)
            
            return image, image_name

             


        




        






        

       


    
            
            
    

   

