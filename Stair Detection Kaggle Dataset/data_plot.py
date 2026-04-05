
import os
import matplotlib.pyplot as plt

import cv2 as cv


def plot_images_with_boxes(image_paths, boxes_list, labels_list=None, figsize=(10, 10), num_images=5):
    """
    Plots images with bounding boxes and optional labels.
    Args:
        image_paths (list of str): List of paths to the images.
        boxes_list (list of list of dict): List of bounding box lists for each image. 
            Each bounding box is a dict with keys 'xmin', 'ymin', 'xmax', 'ymax'.
        labels_list (list of list of str, optional): List of label lists for each image. 
            Each label corresponds to a bounding box. Defaults to None.
        figsize (tuple, optional): Size of the figure. Defaults to (10, 10).
    """
   
    plt.figure(figsize=figsize)
    for i in range(num_images):
        img = cv.imread(image_paths[i])
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)  # Convert BGR to RGB for plotting
        plt.subplot(1, num_images, i + 1)
        plt.imshow(img)
        plt.axis('off')
        boxes = boxes_list[i]
        labels = labels_list[i] if labels_list else [None] * len(boxes)
        for box, label in zip(boxes, labels):
            xmin, ymin, xmax, ymax = box['xmin'], box['ymin'], box['xmax'], box['ymax']
            rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                 fill=False, edgecolor='red', linewidth=2)
            plt.gca().add_patch(rect)
            if label:
                plt.text(xmin, ymin - 5, label, color='red', fontsize=12)
    plt.tight_layout()
    plt.show()