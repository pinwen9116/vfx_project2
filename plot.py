import cv2
import matplotlib.pyplot as plt
import numpy as np


def plot_images(
    images: np.array,
    save_path: str,
    title: str=None
) -> None:
    '''Plot images
    '''
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


def plot_features(
    image: np.array,
    feat_points: np.array,
    save_path: str,
) -> None: 
    img = np.copy(image)

    for feat_pt in feat_points:
        fp_x, fp_y = feat_pt[0], feat_pt[1]
        cv2.circle(img, (fp_x, fp_y), radius=1, color=(0, 255, 255), thickness=4)
    
    cv2.imwrite(save_path, img)


# TODO(yhfang): Try different implementation to plot the orientation of each feature point.
def plot_orientations(
    image: np.array,
    feat_points: np.array,
    orientations: np.array,
    save_path: str,
) -> None: 
    bin_size = 45
    bins = (orientations - (bin_size / 2)) // bin_size
    bins = bins.astype(int)

    bin2coord = {
        0: (1, 0),
        1: (1, -1),
        2: (0, -1),
        3: (-1, -1),
        4: (-1, 0),
        5: (-1, 1),
        6: (0, 1),
        7: (1, 1),
    }

    img = np.copy(image)
    h, w = bins.shape

    for feat_pt in feat_points:
        fp_x, fp_y = feat_pt[0], feat_pt[1]
        dx, dy = bin2coord[bins[fp_y, fp_x]]
        cv2.arrowedLine(img, (fp_x, fp_y), (fp_x + dx, fp_y + dy), color=(0, 0, 255), thickness=2)  
    
    cv2.imwrite(save_path, img)
