import os
import cv2
import numpy as np
import math

from plot import *


class Matching():
    def __init__(
            self,
            images: np.array,
            focal_len: float
        ) -> None:
        self.n_images = images.shape[0]
        self.h = images.shape[1]
        self.w = images.shape[2]
        self.focal_len = focal_len
        self.images = self.warping(images) # shape: (17, 512, 384, 3)
    
    # TODO: Fix color
    def warping(self, images):
        warped_images = np.zeros(images.shape, 'uint8')

        for new_y in range(self.h):
            for new_x in range(self.w):
                x = round(self.focal_len * math.tan((new_x) / self.focal_len))
                y = round(math.sqrt(x**2 + self.focal_len ** 2) * (new_y) / self.focal_len)
                
                if (0 <= x) and (x < self.w) and (0 <= y) and (y < self.h):
                    warped_images[: , new_y, new_x, :] = images[: , y, x, ::-1]
        
        # Visualization
        plot_image(warped_images[:5], './test_data/visualization/warp_images.jpg')

        return warped_images
    
    
    def detection(self):
        ## Harris detector
        sigma = 3
        w = 5
        k = 0.05
        images_R = []
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
            ## gray scale
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            gray = np.float32(gray)

            ## get gradient
            I = cv2.GaussianBlur(gray, (w, w), sigma)
            Ix, Iy = cv2.Sobel(I, cv2.CV_64F, 1, 0, w), cv2.Sobel(I, cv2.CV_64F, 0, 1, w)
            
            ## computer corner response
            Ix2 = Ix ** 2
            Iy2 = Iy ** 2
            Ixy = Ix * Iy
            Sx2 = cv2.GaussianBlur(Ix2, (w, w), sigma)
            Sy2 = cv2.GaussianBlur(Iy2, (w, w), sigma)
            Sxy = cv2.GaussianBlur(Ixy, (w, w), sigma)

            R = (Sx2 * Sy2 - Sxy ** 2) - k * (Sx2+Sy2) ** 2  # (h, w)

            feat_pts_x, feat_pts_y = self._find_feat_points(R, np.mean(R) + 0.5 * np.std(R))
            print(feat_pts_x.shape, feat_pts_y.shape)

    def _find_feat_points(
        self, 
        response: np.array, 
        thres: float,
    ):
        '''Find feature points satisfying:
        1) Corner response is higher than the threshold.
        2) Corner response is higher that that of it\'s neighbor\'s

        Args:
            response (np.array): Harris corner response. Shape: (h, w).
            threshold

        Return:
            feat_points_x (np.array), feat_points_y (np.array)
        '''
        feat_points = np.zeros(response.shape, np.uint8)
        feat_points[response >= thres] = 1

        for y in range(3):
            for x in range(3):
                if x == 1 and y == 1:
                    continue
                kernel = np.zeros((3, 3))
                kernel[1, 1] = 1
                kernel[y, x] = -1

                filtered = np.sign(cv2.filter2D(response, -1, kernel))
                filtered[filtered < 0] = 0
                feat_points &= np.uint8(filtered)

        feat_points = np.where(feat_points == 1)

        return feat_points[1], feat_points[0] 
