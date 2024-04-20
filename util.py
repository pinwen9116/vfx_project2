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
    
    
    def detection(self):
        ## Harris detector
        sigma = 3
        w = 5
        threshold = 0.5            ## might need to change
        k = 0.05

        # Sobel kernels
        Sx = np.array([
            [1, 0, -1],
            [2, 0, -2],
            [1, 0, -1]])

        Sy = Sx.T

        # Gaussian Kernel
        G = np.array([
            [1, 2, 1],
            [2, 4, 2],
            [1, 2, 1]])/16
        
        for img in self.images:

            gray = cv2.cvtColor(self.images, cv2.COLOR_RGB2GRAY)
            gray = np.float32(gray)
            I = cv2.GaussianBlur(gray, (w, w), sigma)
            ## get gradient
            Ix, Iy = cv2.Sobel(I, cv2.CV_64F, 1, 0, w), cv2.Sobel(I, cv2.CV_64F, 0, 1, w)
            
            Ix2 = Ix ** 2
            Iy2 = Iy ** 2
            Ixy = Ix * Iy
            Sx2 = cv2.GaussianBlur(Ix2, (w, w), sigma)
            Sy2 = cv2.GaussianBlur(Iy2, (w, w), sigma)
            Sxy = cv2.GaussianBlur(Ixy, (w, w), sigma)

            ## computer corner response
            R = (Sx2 * Sy2 - Sxy ** 2) - k * (Sx2+Sy2) ** 2

            ## for debug
            print(R)


    