import numpy as np
import cv2 as cv

# The initial camera configuration

K = np.array([[1261, 0, 827],
              [0, 1261, 540],
              [0,   0,  1.]])
img_w, img_h = int(K[0][2]*2), int(K[1][2]*2)
dist_coeff = np.array([-0.2, 0.1, 0, 0])
grid_x, grid_y, grid_z = (-18, 19), (-15, 16), 20
cam_type = 'KB'

obj_pts = np.array([[x, y, grid_z] for y in range(*grid_y) for x in range(*grid_x)], dtype=np.float32)
obj_pts = obj_pts.reshape(-1, 1, 3) if cam_type == 'KB' else obj_pts

while True:
    # Project 3D points with/without distortion
    if cam_type == 'KB':
        dist_pts, _ = cv.fisheye.projectPoints(obj_pts, np.zeros(3), np.zeros(3), K, dist_coeff)
        zero_pts, _ = cv.fisheye.projectPoints(obj_pts, np.zeros(3), np.zeros(3), K, np.zeros(4))
    elif cam_type == 'BC':
        dist_pts, _ = cv.projectPoints(obj_pts, np.zeros(3), np.zeros(3), K, dist_coeff)
        zero_pts, _ = cv.projectPoints(obj_pts, np.zeros(3), np.zeros(3), K, np.zeros(4))

    # Draw vectors
    img_vector = np.full((img_h, img_w, 3), 255, dtype=np.uint8)
    for zero_pt, dist_pt in zip(zero_pts, dist_pts):
        cv.line(img_vector, np.int32(zero_pt.flatten()), np.int32(dist_pt.flatten()), (255, 0, 0))
    for pt in dist_pts:
        cv.circle(img_vector, np.int32(pt.flatten()), 2, (0, 0, 255), -1)

    # Draw grids
    img_grid = np.full((img_h, img_w, 3), 255, dtype=np.uint8)
    dist_pts = dist_pts.reshape(len(range(*grid_y)), -1, 2)
    for pts in dist_pts:
        cv.polylines(img_grid, [np.int32(pts)], False, (0, 0, 255))
    for pts in dist_pts.swapaxes(0, 1):
        cv.polylines(img_grid, [np.int32(pts)], False, (0, 0, 255))

    # Show all images and process key event
    merge = np.hstack((img_vector, img_grid))
    info = f'Focal: {K[0, 0]:.0f}, k1: {dist_coeff[0]:.2f}, k2: {dist_coeff[1]:.2f}, p1: {dist_coeff[2]:.2f}, p2: {dist_coeff[3]:.2f}'
    cv.putText(merge, info, (10, 25), cv.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0))
    cv.imshow('Geometric Distortion Visualization: Vectors | Grids', merge)
    key = cv.waitKey()
    if key == 27: # ESC
        break
    elif key == ord(')') or key == ord('0'):
        K[0,0] += 100
        K[1,1] += 100
    elif key == ord('(') or key == ord('9'):
        K[0,0] -= 100
        K[1,1] -= 100
    elif key == ord('+') or key == ord('='):
        dist_coeff[0] += 0.05
    elif key == ord('-') or key == ord('_'):
        dist_coeff[0] -= 0.05
    elif key == ord(']') or key == ord('}'):
        dist_coeff[1] += 0.05
    elif key == ord('[') or key == ord('{'):
        dist_coeff[1] -= 0.05
    elif key == ord('"') or key == ord("'"):
        dist_coeff[2] += 0.01
    elif key == ord(':') or key == ord(';'):
        dist_coeff[2] -= 0.01
    elif key == ord('>') or key == ord('.'):
        dist_coeff[3] += 0.01
    elif key == ord('<') or key == ord(','):
        dist_coeff[3] -= 0.01

cv.destroyAllWindows()