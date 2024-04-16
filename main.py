import os
import glob

import cv2
import numpy as np
import pandas as pd
from argparse import ArgumentParser

from util import Matching

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--root', type=str, default='../data/original/scene2')
    parser.add_argument('--result_path', type=str, default='../result')
    parser.add_argument("--focal_len", type=float, default=801.63)
    args = parser.parse_args()
    return args

def load_images(root, ann_path):
    '''Get images and corresponding shutter times according to the annotation file.

    Args:
        root: the image root.
        ann_path: the annotation file path.

    Returns:
        `images` and `shutter_times`.
    '''
    print('Loading images...')

    images = []
    for i in range(5):
        image_path = root + "/image_{i}.jpg"
        image = cv2.imread(image_path)
        images.append(image)

    return np.array(images)



def main(args):
    images = cv2.imread(args.root)
    os.makedirs(args.result_path, exist_ok = True)
    
    matched_images = Matching(images=images, focal_len=args.focal_len)
    
    # feature matching:

    # image matching**:

    # bundle adjustment:

    # blending:

    
if __name__ == '__main__':
    main(parse_arguments())