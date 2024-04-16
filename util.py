import os
import cv2
import numpy as np

class Matching():
    def __init__(
            self,
            images: np.array,
            focal_len: float
            ) -> None:
        
        self.images = images
        self.n_images = self.images.shape[0]
        self.h = self.images.shape[1]
        self.w = self.images.shape[2]
        self.focal_len = focal_len
        
    def warping(self):
        