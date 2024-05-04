import os
import glob

import cv2
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from tqdm.auto import tqdm

from util import Matching

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--root', type=str, default='images')
    parser.add_argument('--result_path', type=str, default='../result')
    parser.add_argument("--focal_len", type=float, default=1800)
    parser.add_argument('--compute_homomat', action='store_true')
    args = parser.parse_args()
    return args

def load_images(root):
    '''Get images and corresponding shutter times according to the annotation file.

    Args:
        root: the image root.
        ann_path: the annotation file path.

    Returns:
        `images` and `shutter_times`.
    '''
    print('Loading images...')

    images = []
    for i in range(3):
        image_path = root + f"/IMG_0{103+i}.JPG"
        image = cv2.imread(image_path)
        image = cv2.resize(image, (870, 580), interpolation=cv2.INTER_AREA)

        images.append(image)
    
    # for i in range(16, -1, -1):
    #     image_path = root + f"/image_{i}.jpg"
    #     image = cv2.imread(image_path)

    #     images.append(image)

    return np.array(images)


def main(args):
    images = load_images(args.root)
    print(images.shape)
    os.makedirs(args.result_path, exist_ok = True)
    
    print(f'args.compute_homomat: {args.compute_homomat}')
    match = Matching(images=images, focal_len=args.focal_len, use_ransac_homo=args.compute_homomat)
    
    # feature detection:
    feat_point_list, descriptor_list = match.detection()

    # feature matching:
    coord_pairs = match.feature_match(feat_point_list, descriptor_list)

    # image matching:
    homo_matrices = match.image_matching(coord_pairs)

    # blending:
    panorama = match.blending(homo_matrices)

    # bundle adjustment:

    
if __name__ == '__main__':
    main(parse_arguments())