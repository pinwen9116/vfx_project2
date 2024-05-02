import os
import cv2
import numpy as np
import math
import random

from plot import *
from scipy.spatial.distance import cdist
from tqdm.auto import tqdm


class Matching():
    def __init__(
            self,
            images: np.array,
            focal_len: float,
            k_size: int=5,
            corner_resp_weight: float=1,  # Modify this attribute to control the number of feature points. Higher value leads to fewer points.
            n_bins: float=8,
            p_size: int=8,
            desc_thres: float=0.2,
            match_dist_ratio: float=0.8,
            visualization: bool=True,
        ) -> None:
        self.n_images = images.shape[0]
        self.h = images.shape[1]
        self.w = images.shape[2]
        self.focal_len = focal_len

        self._k_size = k_size
        self._corner_resp_weight = corner_resp_weight
        self._n_bins = n_bins
        self._p_size = p_size
        self._desc_thres = desc_thres
        self._match_dist_ratio = match_dist_ratio

        self.visualization = visualization
        self.images = self.warping(images) # shape: (17, 512, 384, 3)

    def warping(self, images):
        warped_images = np.zeros(images.shape, 'uint8')

        for new_y in range(self.h):
            for new_x in range(self.w):
                x = round(self.focal_len * math.tan((new_x) / self.focal_len))
                y = round(math.sqrt(x**2 + self.focal_len ** 2) * (new_y) / self.focal_len)
                
                if (0 <= x) and (x < self.w) and (0 <= y) and (y < self.h):
                    warped_images[: , new_y, new_x, :] = images[: , y, x, ::-1]
        
        # Visualization
        if self.visualization:
            plot_images(warped_images[:5], './test_data/visualization/warp_images.jpg')

        return warped_images

    def detection(self):
        print('Feature detection...')
        ## Harris detector
        sigma = 3
        k = 0.05

        feat_point_list = list()
        orientation_list = list()
        descriptor_list = list()
        
        for img in tqdm(self.images):
            ## gray scale
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            gray = np.float32(gray)

            ## get gradient
            Ix, Iy = self._compute_gradient(gray)
            
            ## computer corner response
            Ix2 = Ix ** 2
            Iy2 = Iy ** 2
            Ixy = Ix * Iy
            Sx2 = cv2.GaussianBlur(Ix2, (self._k_size, self._k_size), sigma)
            Sy2 = cv2.GaussianBlur(Iy2, (self._k_size, self._k_size), sigma)
            Sxy = cv2.GaussianBlur(Ixy, (self._k_size, self._k_size), sigma)

            R = (Sx2 * Sy2 - Sxy ** 2) - k * (Sx2+Sy2) ** 2  # (h, w)
            
            # Feature points selection: Apply threshold and non-maximum suppression.
            feat_points = self._feat_points_selection(R, np.mean(R) + self._corner_resp_weight * np.std(R))
            
            # Orientation assignment
            orientations = self._compute_orientations(gray, return_orientation=True)

            # Local descriptors
            feat_points, descriptors = self._compute_descriptors(gray, feat_points, orientations)

            feat_point_list.append(feat_points)
            orientation_list.append(orientations)
            descriptor_list.append(descriptors)
        
        if self.visualization:
            plot_features(self.images[0], feat_point_list[0], './test_data/visualization/feature_points.jpg')
            plot_orientations(self.images[0], feat_point_list[0], orientation_list[0], './test_data/visualization/feature_orientations.jpg')

        return feat_point_list, descriptor_list
    
    def feature_match(
        self,
        feat_point_list: list,
        descriptor_list: list,
    ) -> list:
        '''Implementation of feature match.
        
        Args:
            feat_point_list (list): A list of feature points of each image. Shape: (n_image, n_points, 2).
            descriptor_list (list): A list of feature descriptors corresponding to each feature points of each image. Shape: (n_image, n_points, 128).        
        Return:
            coord_pairs (list): A list of matched feature points. [n_pairs, 2 (n_image), n_matches, 2 (x, y)]
        '''
        print('Feature matching...')

        coord_pairs = list()
        for i in tqdm(range(self.n_images - 1)):
            coords_1, coords_2 = self._match_two_images(self.images[i: i+2], feat_point_list[i: i+2], descriptor_list[i: i+2])
            coord_pairs.append([coords_1, coords_2])
        
        return coord_pairs

    def _match_two_images(
        self,
        images: np.array,
        feat_points: list,
        descriptors: list,
    ) -> tuple:
        '''Match feature points between two image. The distance ratio between two descriptors should be less than the given ratio.
        
        Args:
            feat_point_list (list): A list of feature points of each image. Shape: (2, n_points, 2).
            descriptor_list (list): A list of feature descriptors corresponding to each feature points of each image. Shape: (2, n_points, 128).        
        Return:
            coords_1 (list): The list of matched feature point coordinates in the first image. Shape: (n_match, 2).
            coords_2 (list): The list of marched feature point coordinates in the second image. Shape: (n_match, 2).
        '''
        image_1, image_2 = images[0], images[1]
        feat_points_1, feat_points_2 = feat_points[0], feat_points[1]
        descriptors_1, descriptors_2 = descriptors[0], descriptors[1]

        dists = cdist(descriptors_1, descriptors_2, 'euclidean')  # shape: (n_feat_points_1, n_feat_points_2)
        arg_dists = np.argsort(dists, axis=1)

        coords_1 = list()
        coords_2 = list()
        for i, arg_dist in enumerate(arg_dists):
            j1 = arg_dist[0]
            j2 = arg_dist[1]

            if  dists[i, j1] / dists[i, j2] < self._match_dist_ratio:
                coords_1.append(feat_points_1[i])
                coords_2.append(feat_points_2[j1])
        
        if self.visualization:
            plot_feature_match(image_2, coords_2, image_1, coords_1, './test_data/visualization/feature_match.jpg')

        return coords_1, coords_2

    def _feat_points_selection(
        self, 
        response: np.array, 
        thres: float,
    ) -> tuple:
        '''Find feature points satisfying:
        1) Corner response is higher than the threshold.
        2) Corner response is higher that that of it\'s neighbor\'s

        Args:
            response (np.array): Harris corner response. Shape: (h, w).
            thres (float): The corner response threshold for feature points selection.

        Return:
            feat_points (tuple): A tuple of feature points with format (y-coordinates, x-coordinates)
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

        return feat_points

    def _compute_orientations(
        self,
        image: np.array,
        sigma: float=1.5,
        return_orientation: bool=True,
    ) -> np.array:
        '''Implementation of SIFT orientation assignment.

        Args:
            image (np.array): Image with shape (h, w)
            sigma (float): Sigma for GaussianBlur filter.
            return_orientation (bool): `True` for returning orientation; `False` for returning histogram.
        Return:
            Orientation (np.array, shape: (h, w)) or Histogram (np.array, shape: (h, w)).
        '''
        Ix, Iy = self._compute_gradient(image)

        # Orientation assignment
        ori_weights = np.sqrt(Ix ** 2 + Iy ** 2)

        theta = np.rad2deg(np.arctan2(Iy, Ix))
        theta[theta < 0] += 360

        bin_size = 360.0 / self._n_bins
        theta2bin = theta // bin_size

        histogram = np.zeros((self._n_bins, self.h, self.w))

        for b in range(self._n_bins):
            histogram[b][theta2bin == b] = 1
            histogram[b] = cv2.GaussianBlur(histogram[b] * ori_weights, (self._k_size, self._k_size), sigma)
        
        if return_orientation:
            return np.argmax(histogram, axis=0) * bin_size + (bin_size / 2)
        else:
            return histogram

    def _compute_descriptors(
        self,
        gray_image: np.array,
        feat_points: np.array,
        orientations: np.array,
    ) -> tuple:
        '''Implementation of SIFT feature descriptor.

        Args:
            gray_image (np.array): The gray-scaled image. Shape: (h, w).
            feat_points (tuple): A tuple of feature points with format (y-coordinates, x-coordinates).
            orientations (np.array): The orientation of each pixel. Shape: (h, w).

        Returns:
            new_feat_pts (np.array): The feature points [[x, y], ] with valid descriptors. Shape: (n_pts, 2).
            descriptors (np.array): Feature descriptors. Shape: (n_pts, n_bins * (p_size // 2) ** 2)
        '''
        feat_points_y, feat_points_x = feat_points

        descriptors = list()
        new_feat_pts = list()

        for fp_x, fp_y in zip(feat_points_x, feat_points_y):
            if fp_x - self._p_size < 0 or fp_x + self._p_size >= self.w or fp_y - self._p_size < 0 or fp_y + self._p_size >= self.w:
                continue

            rotated_image = self._rotate_image(gray_image, (float(fp_x), float(fp_y)), orientations[fp_y, fp_x])
            rotated_histogram = self._compute_orientations(rotated_image, sigma=self._p_size * 0.5, return_orientation=False,)

            descriptor = list()
            for window_x in range(fp_x - self._p_size, fp_x + self._p_size, self._p_size // 2):
                for window_y in range(fp_y - self._p_size, fp_y + self._p_size, self._p_size // 2):
                    descriptor.extend([np.sum(rotated_histogram[b][window_y: window_y+self._p_size, window_x: window_x+self._p_size]) for b in range(rotated_histogram.shape[0])])
            
            descriptor = self._norm(descriptor)
            descriptor[descriptor > self._desc_thres] = self._desc_thres

            descriptors.append(self._norm(descriptor))
            new_feat_pts.append([fp_x, fp_y])

        return np.array(new_feat_pts), np.array(descriptors)

    def _compute_gradient(
        self,
        image: np.array,
        sigma: float=3,
    ):
        I = cv2.GaussianBlur(image, (self._k_size, self._k_size), sigma)
        Ix, Iy = cv2.Sobel(I, cv2.CV_64F, 1, 0, self._k_size), cv2.Sobel(I, cv2.CV_64F, 0, 1, self._k_size)

        return Ix, Iy
    
    def _rotate_image(
        self,
        image: np.array,
        center: tuple,
        theta: float,
    ):
        '''Rotate image with regard to the center.

        Args: 
            image (np.array): Image to be rotated.
            center (tuple): Rotation center (x, y).
            theta (float): Degree of rotation.
        
        Return:
            The image after rotation.
        '''
        matrix = cv2.getRotationMatrix2D(center, theta, 1)

        return cv2.warpAffine(image, matrix, (self.w, self.h))
    
    def _norm(
        self,
        vector: np.array,
    ):
        norm = np.linalg.norm(vector)
        return vector / norm if norm != 0 else vector
    
    def RANSAC(
            self,
            coor_pair,
            n_sample,
            n_iter,
            n_subSample,
            threshold
    ):
        '''Implementation of RANSAC

        Args:
            coor_pair (list): The coordinates of the matching pairs between two images. `coor_pair[0]` represents the points in the first image, while `coor_pair[0]` represents that in the second one.
            n_sample (int): Number of matching pairs. This should be equal to `len(coor_pair[0])` and `len(coor_pair[1])`.
            n_iter (int): The number of iteration.
            n_subSample (int): The number of samples to formulate the fitted model.
            threshold (float): 

        Return:
            H (np.array): The homography matrix
        '''
        coor_1 = np.array(coor_pair[0])
        coor_2 = np.array(coor_pair[1])

        max_in = 0
        best_H = None
        for _ in range(n_iter):
            # Draw `n_subSample` points and fit the model with them.
            subSampleIdx = random.sample(range(n_sample), n_subSample)  # (n_subSample, 2)
            H = self.homography(coor_1[subSampleIdx], coor_2[subSampleIdx])
            print(f'H.shape: {H.shape}')  # (3, 3)
            num_in = 0

            for idx in range(n_sample):
                if idx not in subSampleIdx:
                    concate_coor = np.hstack((coor_1[idx], coor_2[idx]))
                    print(f'concate_coor.shape: {concate_coor.shape}') # (4, )
                    dst_coor = H @ concate_coor.T
                    if dst_coor[2] <= 1e-5:      ## avoid 0 division
                        continue
                    # Q: 為什麼要 /= 2?
                    dst_coor /= 2
                    # 為什麼是這兩個相減，應該要用到 `dst_coor` 才對吧
                    if (np.linalg.norm(coor_pair[:2][1] - coor_pair[idx][1]) < threshold):
                        num_in += 1
            if max_in <= num_in:
                max_in = num_in
                best_H = H

        return best_H

    # TODO: Check the formula and shape of H.
    def homography(
            self,
            P,
            m
    ):
        # solve homography matrix
        A = []
        for r in range(len(P)):
            # print(m[r, 0])      ## for debug
            A.append([-P[r, 0], -P[r, 1], -1, 0, 0, 0, P[r, 0]*m[r,0], P[r,1]*m[r,0], m[r, 0]])
            A.append([0, 0, 0, -P[r,0], -P[r,1], -1, P[r,0]*m[r,1], P[r,1]*m[r,1], m[r,1]])
        u, s, v = np.linalg.svd(A)
        H = np.reshape(v[8], (3, 3))            
        ## normalize
        H = (1/H.item(8)) * H
        return H        
    
    def blending(
            self,
            homomat
    ):
        res = self.images[0]
        inv_H = np.linalg.inv(homomat)
        for i in range(1, self.n_images):
            left_img = res
            right_img = self.images[i]
            (lh, lw) = left_img[:2]
            (rh, rw) = right_img[:2]
            stitch_res = np.zeros((max(lh, rh), lw+rw, 3), dtype='int')
            for i in range(stitch_res.shape[0]):
                for j in range(stitch_res.shape[1]):
                    coor = np.array([j, i, 1])
                    res_img_coor = inv_H @ coor
                    res_img_coor /= res_img_coor[2]
                    ## TODO: knn or  bilinear interpolation
                    y, x = int(round(res_img_coor[0])), int(round(res_img_coor[1]))
                    if x < 0 or x >= rh or y < 0 or y >= rw:
                        continue
                    stitch_res[i, j] = right_img[x, y]

        ## TODO: blending