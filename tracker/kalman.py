from filterpy.kalman import KalmanFilter
import numpy as np
from enum import Enum

class TrackStatus(Enum):
    Tentative = 0
    Confirmed = 1
    Coasted   = 2

class KalmanTracker(object):

    count = 1

    def __init__(self, y, R, wx, wy, vmax, w, h, dt=1/30):
        
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        # state transistion matrix
        self.kf.F = np.array([[1, dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, dt], [0, 0, 0, 1]])
        # measurement function
        self.kf.H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
        # measurement uncertainty/noise
        self.kf.R = R
        # covariance matrix
        self.kf.P = np.zeros((4, 4))
        np.fill_diagonal(self.kf.P, np.array([1, vmax**2/3.0, 1,  vmax**2/3.0]))

        # Process noise Q
        G = np.zeros((4, 2))
        G[0,0] = 0.5*dt*dt
        G[1,0] = dt
        G[2,1] = 0.5*dt*dt
        G[3,1] = dt
        Q0 = np.array([[wx, 0], [0, wy]])
        # Qk = G·diag(σx,σy)·G^T
        self.kf.Q = np.dot(np.dot(G, Q0), G.T)

        # filter state estimate
        self.kf.x[0] = y[0]
        self.kf.x[1] = 0
        self.kf.x[2] = y[1]
        self.kf.x[3] = 0

        self.id = KalmanTracker.count
        KalmanTracker.count += 1
        self.age = 0
        self.death_count = 0
        self.birth_count = 0
        self.detidx = -1
        self.w = w
        self.h = h

        self.status = TrackStatus.Tentative


    def update(self, y, R):
        self.kf.update(y,R)

    def predict(self):
        self.kf.predict()
        self.age += 1
        return np.dot(self.kf.H, self.kf.x)

    def get_state(self):
        return self.kf.x
    
    def distance(self, y, R):
        # Calculate Residual:
        diff = y - np.dot(self.kf.H, self.kf.x)
        # Compute Residual Covariance Matrix:
        S = np.dot(self.kf.H, np.dot(self.kf.P,self.kf.H.T)) + R
        # Calculate Normalized Mahalanobis Distance:
        SI = np.linalg.inv(S) # S^-1
        mahalanobis = np.dot(diff.T,np.dot(SI,diff))
        logdet = np.log(np.linalg.det(S)) # ln|S|
        return mahalanobis[0,0] + logdet