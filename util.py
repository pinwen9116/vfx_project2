import os
import cv2
import numpy as np
import math

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
        warped_images = np.zeros(self.h, self.w, self.n_images, 'uint8')
        x0 = self.w / 2
        y0 = self.h / 2
        for new_y in range(self.h):
            for new_x in range(self.w):
                x = round(x0 + self.focal_len*math.tan(new_x - x0)/self.focal_len)
                y = round(y0 + math.sqrt(x**2 + self.focal_len ** 2) * (new_y - y0) / self.focal_len)
                if (0 < x) and (x <= self.w) and (0 < y) and (y <= self.h):
                    warped_images[new_y, new_x, :, :] = self.images[y, x, :, :]
                else:
                    warped_images[new_y, new_x, :, :] = 0
        return warped_images
    