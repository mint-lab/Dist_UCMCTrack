import numpy as np
import os
import cv2 as cv
import torch
from scipy.spatial.transform import Rotation as R


def getUVError(box):
    # ????????
    u = 0.05*box[3]
    v = 0.05*box[3]
    if u>13:
        u = 13
    elif u<2:
        u = 2
    if v>10:
        v = 10
    elif v<2:
        v = 2
    return u,v
    

def parseToMatrix(data, rows, cols):
    matrix_data = np.fromstring(data, sep=' ')
    matrix_data = matrix_data.reshape((rows, cols))
    return matrix_data

def readKittiCalib(filename):
    # 检查文件是否存在
    if not os.path.isfile(filename):
        print(f"Calib file could not be opened: {filename}")
        return None,False

    P2 = np.zeros((3, 4))
    R_rect = np.identity(4)
    Tr_velo_cam = np.identity(4)
    KiKo = None

    with open(filename, 'r') as infile:
        for line in infile:
            id, data = line.split(' ', 1)
            if id == "P2:":
                P2 = parseToMatrix(data, 3, 4)
            elif id == "R_rect":
                R_rect[:3, :3] = parseToMatrix(data, 3, 3)
            elif id == "Tr_velo_cam":
                Tr_velo_cam[:3, :4] = parseToMatrix(data, 3, 4)
            KiKo = np.dot(np.dot(P2, R_rect), Tr_velo_cam)

    return KiKo, True

def readCamParaFile(camera_para):
    R = np.zeros((3, 3))
    T = np.zeros((3, 1))
    dist = np.zeros(4)
    IntrinsicMatrix = np.zeros((3, 3))

    try:
        with open(camera_para, 'r') as f_in:
            lines = f_in.readlines()
        i = 0
        while i < len(lines):
            if lines[i].strip() == "RotationMatrices":
                i += 1
                for j in range(3):
                    R[j] = np.array(list(map(float, lines[i].split())))
                    i += 1
            elif lines[i].strip() == "TranslationVectors":
                i += 1
                T = np.array(list(map(float, lines[i].split()))).reshape(-1,1)
                T = T / 1000
                i += 1
            elif lines[i].strip() == "IntrinsicMatrix":
                i += 1
                for j in range(3):
                    IntrinsicMatrix[j] = np.array(list(map(float, lines[i].split())))
                    i += 1
            elif lines[i].strip() == "Distortion":
                i += 1
                dist = np.array(list(map(float, lines[i].split())))
                i += 1

            else:
                i += 1
    except FileNotFoundError:
        print(f"Error! {camera_para} doesn't exist.")
        return None, None, None, None, False

    return IntrinsicMatrix, R, T, dist, True

class Mapper(object):
    def __init__(self, campara_file,dataset= "kitti"):
        if dataset == "kitti":
            self.KiKo, self.is_ok = readKittiCalib(campara_file)
            z0 = -1.73
        else:
            self.K, self.R, self.T, self.dist, self.is_ok = readCamParaFile(campara_file)           

    def localize_point(self, uv, K, distort=None, R=np.eye(3), T=np.zeros((3, 1)), polygons={}, planes=[]):
        '''Calculate 3D location (unit: [meter]) of the given point (unit: [pixel]) with the given camera configuration'''
        # Make a ray aligned to the world coordinate
        ori = R.T
        pos = -R.T @ T.squeeze()

        uv = np.array(uv).reshape(-1, 1, 2).astype(np.float32)
        uv_undistort = cv.fisheye.undistortPoints(uv, K, distort).flatten()
        # uv_undistort = cv.undistortPoints(uv, K, distort).flatten()
        r = ori @ np.append(uv_undistort, 1) # A ray with respect to the world coordinate
        scale = np.linalg.norm(r)
        r = r / scale

        # Get a plane if 'pt' exists inside of any 'polygons'
        n, d = np.array([0, 0, 1]), 0

        # Calculate distance and position on the plane
        denom = n.T @ r
        # if np.fabs(denom) < 1e-6: # If the ray 'r' is almost orthogonal to the plane norm 'n' (~ almost parallel to the plane)
        #     return None, None
        distance = -(n.T @ pos + d) / denom
        # r_c = ori.T @ (np.sign(distance) * r)
        # if r_c[-1] <= 0: # If the ray 'r' stretches in the negative direction (negative Z)
        #     return None, None
        # X = Camera  position + k * ray
        xy = pos + distance * r

        return xy[0:2].reshape(2,1)

    def unscented_transform_point(self, uv, sigma_uv, K, distort=None, R=np.eye(3), T=np.zeros((3, 1))):
        # https://dibyendu-biswas.medium.com/extended-kalman-filter-a5c3a41b2f80
        n = uv.shape[0]  # Dimension of the input (2D point)

        # Calculate lambda
        lambda_ = 3 - n

        # Create sigma points
        sigma_points = np.zeros((n * 2 + 1, n))
        sigma_points[0] = uv.flatten()

        sqrt_cov = np.sqrt((n + lambda_) * sigma_uv)
        for i in range(n):
            sigma_points[i + 1] = uv.flatten() + sqrt_cov[i]
            sigma_points[n + i + 1] = uv.flatten() - sqrt_cov[i]

        # Transform sigma points using the localization function
        transformed_points = np.array([self.localize_point(sig_point, K, distort, R, T).flatten() for sig_point in sigma_points])

        # Calculate new mean and covariance
        weights_mean = np.full((2 * n + 1,), 1 / (2 * (n + lambda_)))
        weights_mean[0] = lambda_ / (n + lambda_)
        
        new_mean = np.dot(weights_mean, transformed_points)

        # Calculate new covariance
        covariance_new = np.zeros((2, 2))
        for i in range(2 * n + 1):
            diff = transformed_points[i] - new_mean
            covariance_new += weights_mean[i] * np.outer(diff, diff)

        return covariance_new

    def uv2xy(self, uv, sigma_uv):
        xy = self.localize_point(uv, self.K, self.dist, self.R, self.T)
        sigma_xy = self.unscented_transform_point(uv, sigma_uv, self.K, self.dist, self.R, self.T)
        return xy, sigma_xy
    
    def xy2uv(self, x, y):
        '''Convert the given 3D point to pixel coordinates with the given camera configuration'''
        
        # Squeeze the position vector
        points_3D = np.array([[x, y, 0]], dtype='float32') 

        rvec, _ = cv.Rodrigues(self.R)
        tvec = self.T.squeeze()

        points_3D = points_3D.reshape(-1, 1, 3)
        uv, _ = cv.fisheye.projectPoints(points_3D, rvec, tvec, self.K, self.dist)
        # uv, _ = cv.projectPoints(points_3D, rvec, tvec, self.K, self.dist)
        
        return uv[0][0][0], uv[0][0][1]

    def mapto(self,box):
        # box [x, y, w, h]
        # (x, y) top left conner cordinate
        # w: width, h: height
        uv = np.array([[box[0]+box[2]/2], [box[1]+box[3]]])
        u_err,v_err = getUVError(box)
        sigma_uv = np.identity(2)
        sigma_uv[0,0] = u_err*u_err
        sigma_uv[1,1] = v_err*v_err
        y,R = self.uv2xy(uv, sigma_uv)
        return y,R
    
    # def disturb_campara(self,z):

    #     # 根据z轴旋转，构造旋转矩阵Rz
    #     Rz = np.array([[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]])

    #     R = np.dot(self.Ko[:3, :3],Rz)
    #     # 将self.Ko 拷贝到新变量 Ko_new
    #     Ko_new = self.Ko.copy()
    #     Ko_new[:3, :3] = R
    #     self.KiKo = np.dot(self.Ki, Ko_new)
    #     self.A[:, :2] = self.KiKo[:, :2]
    #     self.A[:, 2] = self.KiKo[:, 3]
    #     self.InvA = np.linalg.inv(self.A)

    # def reset_campara(self):
    #     self.KiKo = np.dot(self.Ki, self.Ko)
    #     self.A[:, :2] = self.KiKo[:, :2]
    #     self.A[:, 2] = self.KiKo[:, 3]
    #     self.InvA = np.linalg.inv(self.A)


