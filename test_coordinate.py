import numpy as np
import open3d as o3d

if __name__ == "__main__":
    # Define rotation matrices
    R_our = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])  # made by ours
    R_tar = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])  # made by ucmc

    # Calculate transformation matrix
    R_tra = R_tar @ R_our.T

    # Print matrices
    print("R_our:\n", R_our)
    print("R_tar:\n", R_tra)
    print("R_tra (calculated transformation matrix):\n", R_tra @ R_our)
    
    
    # R_test     
    R_test = np.array([
        [0.999981425814831, -0.002016116860107875, -0.005751808249948209],
        [-0.006094917992703717, -0.33078040012900267, -0.9436880728636763],
        [0.0, 0.9437056014262623, -0.3307865442195264]
    ])
    
    # Define R_gt (ground truth rotation matrix)
    R_gt = np.array([
        [-0.12188, -0.99239, 0.01745],
        [-0.88012, 0.09993, -0.46412],
        [0.45885, -0.07193, -0.88560]
    ])
    
    print("Transformed R_test \n", R_gt @ R_test.T)
    print("R_GT \n ", R_gt)

