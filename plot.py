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
        img = plt.imshow(image[:, :, ::-1])
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
    print(f'Number of feature points: {len(feat_points)}')

    img = np.copy(image[:, :, ::-1])

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

    img = np.copy(image[:, :, ::-1])
    h, w = bins.shape

    for feat_pt in feat_points:
        fp_x, fp_y = feat_pt[0], feat_pt[1]
        dx, dy = bin2coord[bins[fp_y, fp_x]]
        cv2.arrowedLine(img, (fp_x, fp_y), (fp_x + dx, fp_y + dy), color=(0, 0, 255), thickness=2)  
    
    cv2.imwrite(save_path, img)


def plot_feature_match(
    image_1: np.array,
    feat_points_1: list,
    image_2: np.array,
    feat_points_2: list,
    save_path: str,
) -> None: 
    print(f'Number of matching pairs: {len(feat_points_1)}')

    h, w, c = image_1.shape

    img = np.zeros((h, w * 2, c))
    img[:h, :w] = image_1
    img[:h, w:] = image_2

    for feat_pt1, feat_pt2 in zip(feat_points_1, feat_points_2):
        feat_pt2[0] = feat_pt2[0] + w

        feat_pt1 = tuple(feat_pt1)
        feat_pt2 = tuple(feat_pt2)

        cv2.line(img, feat_pt1, feat_pt2, color=(0, 0, 255), thickness=1)
    
    cv2.imwrite(save_path, img[:, :, ::-1])
