import cv2
import matplotlib.pyplot as plt
import numpy as np


def plot_image(images, save_path, title=None):
    n_col = len(images)

    fig = plt.figure(figsize=(15, 5))
    
    for i, image in enumerate(images):
        plt.subplot(1, n_col, i + 1)
        img = plt.imshow(image)
        if title != None:
            plt.title(title)
        plt.axis('off')
    
    plt.tight_layout()
    fig.savefig(save_path)
