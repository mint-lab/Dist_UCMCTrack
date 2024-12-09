import shutil
import cv2 as cv
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import json


def trim_object(obj, image_size):
    '''Trim 2D points within the given image size'''
    image_w, image_h = image_size
    obj[(obj[:,0] < 0), 0] = 0
    obj[(obj[:,1] < 0), 1] = 0
    obj[(obj[:,0] >= image_w), 0] = image_w - 1
    obj[(obj[:,1] >= image_h), 1] = image_h - 1

def readCamParaFile(camera_para, flag_KRT=False):
    R = np.zeros((3, 3))
    T = np.zeros((3, 1))
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
                T = np.array(list(map(float, lines[i].split()))).reshape(-1, 1)
                T = T / 1000
                i += 1
            elif lines[i].strip() == "IntrinsicMatrix":
                i += 1
                for j in range(3):
                    IntrinsicMatrix[j] = np.array(list(map(float, lines[i].split())))
                    i += 1
            else:
                i += 1
    except FileNotFoundError:
        print(f"Error! {camera_para} doesn't exist.")
        return None, False

    Ki = np.zeros((3, 4))
    Ki[:, :3] = IntrinsicMatrix

    Ko = np.eye(4)
    Ko[:3, :3] = R
    Ko[:3, 3] = T.flatten()

    if flag_KRT:
        return IntrinsicMatrix, R, T.flatten(), True
    else:
        KiKo = np.dot(Ki, Ko)
        return Ki, Ko, True

def distort_img(image, f, c, dist_coeffs, cam_type):
    fx, fy = f
    cx, cy = c  

    K = np.array([[fx, 0, cx],
                 [0, fy, cy],
                 [0, 0, 1]])

    K = K.astype(np.float32)
    
    h, w = image.shape[:2]
    u, v = np.meshgrid(np.arange(w), np.arange(h))

    # Normalize coordinates
    uv = np.stack((u, v), axis=-1).reshape(-1, 1, 2)  # 각 점의 (u, v) 좌표 쌍 생성 및 리쉐이프
    uv = uv.astype(np.float32) 

    # Using undistortPoints to calculate distorted coordinates
    if cam_type == 'BC':
        uv_distorted = cv.undistortPoints(uv, K, dist_coeffs)
          # 원래 이미지 형태로 리쉐이프
    elif cam_type == 'KB':
        uv_distorted = cv.fisheye.undistortPoints(uv, K=K, D=dist_coeffs)

    uv_distorted = uv_distorted.reshape(h, w, 2)  # 원래 이미지 형태로 리쉐이프
    
    # 분할
    u_distorted = uv_distorted[:, :, 0]  # u 좌표
    v_distorted = uv_distorted[:, :, 1]  # v 좌표

    # Denormalize the coordinates
    u_distorted = (u_distorted * fx) + cx
    v_distorted = (v_distorted * fy) + cy
    map_x = np.float32(u_distorted)
    map_y = np.float32(v_distorted)
    distorted_image = cv.remap(image, map_x, map_y, interpolation=cv.INTER_LINEAR)
    
    return distorted_image

def distort_points_BC(pts, f, c, dist_coeffs):
    fx, fy = f
    cx, cy = c
    k1, k2, k3, k4  = dist_coeffs
    
    # 변환된 포인트를 저장할 리스트
    distorted_pts = []

    for pt in pts:
        x, y = pt
        
        # 렌즈 중심 기준으로 좌표 정규화
        x_norm = (x - cx) / fx
        y_norm = (y - cy) / fy
        
        # 반경 계산
        r = np.sqrt(x_norm**2 + y_norm**2)
        
        # 방사 왜곡 적용
        x_distorted = x_norm * (1 + k1 * r**2 + k2 * r**4 + k3 * r**6)
        y_distorted = y_norm * (1 + k1 * r**2 + k2 * r**4 + k3 * r**6)
        
        # 원래 픽셀 좌표로 변환
        x_distorted = x_distorted * fx + cx
        y_distorted = y_distorted * fy + cy
        
        distorted_pts.append((x_distorted, y_distorted))
    
    return np.array(distorted_pts)

def distort_points_KB(pts, f, c, dist_coeffs):
    fx, fy = f
    cx, cy = c
    k1, k2, k3, _ = dist_coeffs
    # 변환된 포인트를 저장할 리스트
    distorted_pts = []

    for pt in pts:
        x, y = pt
        
        # 렌즈 중심 기준으로 좌표 정규화
        x_norm = (x - cx) / fx
        y_norm = (y - cy) / fy
        
        # 반경 계산
        r = np.sqrt(x_norm**2 + y_norm**2)
        
        # 각도 계산
        theta = np.arctan(r)
        
        # Kannala-Brandt 모델의 왜곡 적용
        theta_distorted = theta * (1 + k1 * theta**2 + k2 * theta**4)
        
        # 왜곡된 반경을 사용해 정규화된 좌표 계산
        scale = theta_distorted / r if r != 0 else 1  # r이 0인 경우 scale을 1로 설정
        x_distorted = x_norm * scale
        y_distorted = y_norm * scale
        
        # 원래 픽셀 좌표로 변환
        x_distorted = x_distorted * fx + cx
        y_distorted = y_distorted * fy + cy
         
        distorted_pts.append((x_distorted, y_distorted))
    
    return np.array(distorted_pts)
 
def convert_raw_to_pts(bb_left, bb_top, bb_w, bb_h):
    """
    Convert uvwh to each point array
    """
    pt1 = np.array([bb_left,               bb_top])
    pt2 = np.array([bb_left + bb_w,        bb_top])
    pt3 = np.array([bb_left + bb_w, bb_top + bb_h])
    pt4 = np.array([bb_left,        bb_top + bb_h])
    return pt1, pt2, pt3, pt4

def generate_midpoints(pt1, pt2, num_points=10):
    """
    Generate midpoints between two points
    """
    return np.linspace(pt1, pt2, num=num_points, endpoint=False)

def generate_all_midpoints(bb_left, bb_top, bb_w, bb_h, num_points=10):
    """
    Generate 50 points between each pair of points forming the rectangle
    """
    # Get the four corner points
    pt1, pt2, pt3, pt4 = convert_raw_to_pts(bb_left, bb_top, bb_w, bb_h)

    # Generate midpoints for each side
    midpoints1 = generate_midpoints(pt1, pt2, num_points)
    midpoints2 = generate_midpoints(pt2, pt3, num_points)
    midpoints3 = generate_midpoints(pt3, pt4, num_points)
    midpoints4 = generate_midpoints(pt4, pt1, num_points)

    # Combine all midpoints into a single array
    new_pts = np.vstack((midpoints1, midpoints2, midpoints3, midpoints4))
    return new_pts.astype(np.int16)

def test_distort(image_file, output_file, f, c, cam_distort):
    # 이미지 로드
    image = cv.imread(image_file)

    # 왜곡 계수 설정 (양수는 배럴 왜곡, 음수는 핀쿠션 왜곡)
    k1, k2, k3 = cam_distort

    # 왜곡 적용
    distorted_image = distort_img(image, f, c, k1, k2, k3)
    cv.imwrite(output_file, distorted_image)
    print(f"Save distorted image in {output_file}")

    # 결과 이미지 보기
    # GUI 환경에서만 사용 가능
    cv.imshow("Distorted Image", distorted_image)
    cv.destroyAllWindows()

def test_distort_new(image_file, f, c, cam_distort, det_result_file, cam_type):
    # 이미지 로드
    image = cv.imread(image_file)

    # 왜곡 적용
    distorted_image1 = distort_img(image, f, c, cam_distort, cam_type)


    with open(det_result_file) as f_in:
        
        det_results = f_in.read().splitlines()
        for det_result in det_results:
            # e.g) 1,1,584.6,446.2,87.8,261.9,0.96,-1,-1,-1
            frame_id, id, bb_left, bb_top, bb_w, bb_h, confidence_score, x, y, z = det_result.split(',')
            if frame_id == '1':
                u, v, w, h = int(float(bb_left)), int(float(bb_top)), int(float(bb_w)), int(float(bb_h))

                # Plot original bbox 
                cv.rectangle(image, (u, v), ((u + w), (v + h)), (0, 255, 0), 2)
                

                # Distort points 
                pts = generate_all_midpoints(u, v, w, h, num_points=30)  
                # for pt in pts:
                #     cv.circle(image, pt,  2, (255, 0,0), -1)
                
                if cam_type == 'BC':
                    distorted_pts1 = distort_points_BC(pts, f, c, cam_distort) 
                elif cam_type =='KB':
                    distorted_pts1 = distort_points_KB(pts, f, c, cam_distort) 
                    distorted_pts1 = distorted_pts1.reshape(-1, 2)
                # for pt in distorted_pts1:
                #     cv.circle(distorted_image1, pt.astype(np.int16),  2, (0, 0,255), -1)
                

                # Ensure distorted_pts is a valid numpy array
                if distorted_pts1.size > 0:
                    distorted_pts1 = np.array(distorted_pts1, dtype=np.float32) 

                    # Create new bounding box which is smallest rectangle including distorted points  
                    u1, v1, w1, h1 = cv.boundingRect(distorted_pts1)

                    # Draw the rectangle on the image
                    cv.rectangle(distorted_image1, (u1, v1), (u1 + w1, v1 + h1), (0, 0, 255), 2)
        
  
    plt.subplot(2, 2, 1)
    plt.imshow(image)
    plt.axis("off")
    plt.title("Original Image")

    # Vẽ ảnh 2
    plt.subplot(2, 2, 2)
    plt.imshow(distorted_image1)
    plt.axis("off")
    plt.title("Distorted Image")

    plt.show()

def test_dist_pts(image_file, det_result_file, K, cam_distort):
    with open(det_result_file) as f_in:
        image = cv.imread(image_file)
        
        image = distort_new(image, K, cam_distort)
        det_result = f_in.read().splitlines()[0]
        # e.g) 1,1,584.6,446.2,87.8,261.9,0.96,-1,-1,-1
        frame_id, id, bb_left, bb_top, bb_w, bb_h, confidence_score, x, y, z = det_result.split(',')

        # Plot original bbox 
        cv.rectangle(image, (int(float(bb_left)), int(float(bb_top))), 
                     ((int(float(bb_left)) + int(float(bb_w))), (int(float(bb_top))+int(float(bb_h)))), (0, 255, 0), 2)

        # Distort points 
        pts = convert_raw_to_pts(int(float(bb_left)), int(float(bb_top)), int(float(bb_w)), int(float(bb_h)))        
        distorted_pts = distort_points_new(pts, K, cam_distort)

        # Ensure distorted_pts is a valid numpy array
        if distorted_pts.size > 0:
            distorted_pts = np.array(distorted_pts, dtype=np.float32)  # Convert to float32 if necessary
            # Create new bounding box which is smallest rectangle including distorted points  
            u, v, w, h = cv.boundingRect(distorted_pts)
            
            # Draw the rectangle on the image
            cv.rectangle(image, (u, v), (u + w, v + h), (0, 0, 255), 2)
            
            # Display the image with the rectangle
            cv.imshow('Image with Bounding Box', image)
            cv.waitKey(0)
            cv.destroyAllWindows()
        
def create_distorted_mot17(gt_file, save_file, cam_dist, cam_type):
    with open(gt_file) as f_in:
        gts = f_in.read().splitlines()

        os.makedirs(os.path.dirname(save_file), exist_ok=True)
        with open(save_file, 'w') as out_file:
            for gt in tqdm(gts):
                frame_id, id, bb_left, bb_top, bb_w, bb_h, confidence_score, class_id, visibility = gt.split(',')
                # u, v, w, h = int(float(bb_left)), int(float(bb_top)), int(float(bb_w)), int(float(bb_h))
                # pts = convert_raw_to_pts(u, v, w, h)
                u, v, w, h = int(float(bb_left)), int(float(bb_top)), int(float(bb_w)), int(float(bb_h))
                pts = generate_all_midpoints(u, v, w, h, num_points=20)  

                if cam_type == 'BC':
                    distorted_pts = distort_points_BC(pts, f, c, cam_dist)
                elif cam_type == 'KB': 
                    distorted_pts = distort_points_KB(pts, f, c, cam_dist)

                # Ensure distorted_pts is a valid numpy array
                if distorted_pts.size > 0:
                    distorted_pts = np.array(distorted_pts, dtype=np.float32)  # Convert to float32 if necessary
                    
                    # Create new bounding box which is smallest rectangle including distorted points  
                    u, v, w, h = cv.boundingRect(distorted_pts)
                    
                    # Write the results to file
                    out_file.write(f"{frame_id},{id},{u},{v},{w},{h},{confidence_score},{class_id},{visibility}\n")
                else:
                    print(f"Warning: No distorted points found for frame_id {frame_id} and id {id}")

def create_distorted_det_results(det_result_file, save_file, cam_dist, cam_type):
    with open(det_result_file) as f_in:
        det_results = f_in.read().splitlines()
        
        if not os.path.isfile(save_file):
            # Create any missing directories in the path
            os.makedirs(os.path.dirname(save_file), exist_ok=True)

        with open(save_file, 'w') as out_file:
            for det_result in tqdm(det_results):
                # e.g) 1,1,584.6,446.2,87.8,261.9,0.96,-1,-1,-1
                frame_id, id, bb_left, bb_top, bb_w, bb_h, confidence_score, x, y, z = det_result.split(',')

                u, v, w, h = int(float(bb_left)), int(float(bb_top)), int(float(bb_w)), int(float(bb_h))
                pts = generate_all_midpoints(u, v, w, h, num_points=20)  
                
                # Use only first frame image to get width and height of image
                if cam_type == 'BC':
                    distorted_pts = distort_points_BC(pts, f, c, cam_dist)
                elif cam_type == 'KB':
                    distorted_pts = distort_points_KB(pts, f, c, cam_dist)

                # Ensure distorted_pts is a valid numpy array
                if distorted_pts.size > 0:
                    distorted_pts = np.array(distorted_pts, dtype=np.float32)  # Convert to float32 if necessary
                    
                    # Create new bounding box which is smallest rectangle including distorted points  
                    u, v, w, h = cv.boundingRect(distorted_pts)
                    
                    # Write the results to file
                    out_file.write(f"{frame_id},{id},{u},{v},{w},{h},{confidence_score},{x},{y},{z}\n")
                else:
                    print(f"Warning: No distorted points found for frame_id {frame_id} and id {id}")

def create_cam_para(src_cam_para, dst_cam_para, cam_dist):
    # Read the content of the original file
    with open(src_cam_para, 'r') as f:
        lines = f.readlines()
    
    # Convert cam_dist (NumPy array) to a space-separated string
    cam_dist_str = " ".join(map(str, cam_dist.flatten()))
    
    # Write the original content and the new cam_dist line to the new file
    os.makedirs(os.path.dirname(dst_cam_para), exist_ok=True)
    with open(dst_cam_para, 'w') as f:
        f.writelines(lines)  # Write the original lines
        f.write("\nDistortion\n")  # Add a label for cam_dist
        f.write(cam_dist_str + "\n")  # Write cam_dist values as a new line

def copy_file2dir(src_file, dest_dir):
    """
    Copy a file to a destination directory.

    Args:
        source_file (str): The full path of the source file.
        dest_dir (str): The full path of the destination directory.

    Raises:
        FileNotFoundError: If the source file does not exist.
        OSError: If an error occurs during the copy operation.
    """
    # Create the destination directory if it doesn't exist
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # Construct the full path of the destination file
    dest_file = os.path.join(dest_dir, os.path.basename(src_file))

    try:
        # Copy the file
        shutil.copy2(src_file, dest_file)
        print(f"Copied {src_file} to {dest_dir}")
    except FileNotFoundError:
        print(f"File {src_file} does not exist.")
        raise
    except OSError as e:
        print(f"Error copying {src_file} to {dest_dir}: {e}")
        raise


if __name__ == "__main__":

    # new GT 
    sequences = ["MOT17-02-SDP", "MOT17-04-SDP", "MOT17-05-SDP","MOT17-09-SDP",
                 "MOT17-10-SDP","MOT17-11-SDP","MOT17-13-SDP"]
    cam_type = 'KB'
    
    # BC
    # dist_coeffs = [ np.array([-0.45, 0.10, 0.0, 0.0]), # 02
    #                np.array([-0.40, 0.10, 0.0, 0.0]),  # 03
    #                np.array([-0.65, 0.25, 0.0, 0.0]),  # 05
    #                np.array([-0.45, 0.10, 0.0, 0.0]),  # 09
    #                np.array([-0.35, 0.05, 0.0, 0.0]),  # 10
    #                np.array([-0.45, 0.10, 0.0, 0.0]),  # 11
    #                np.array([-0.40, 0.10, 0.0, 0.0])]  # 13

    # KB best
    dist_coeffs = [ np.array([-0.25, 0.00, 0.0, 0.0]),  # 02
                   np.array([-0.15, 0.05, 0.0, 0.0]),  # 04
                   np.array([-0.25, 0.20, 0.0, 0.0]),  # 05
                   np.array([-0.20, 0.10, 0.0, 0.0]),  # 09
                   np.array([-0.15, 0.10, 0.0, 0.0]),  # 10
                   np.array([-0.20, 0.10, 0.0, 0.0]),  # 11
                   np.array([-0.20, 0.05, 0.0, 0.0])]  # 13

    
    for seq, dist_coeff  in zip(sequences, dist_coeffs):    

        # Get Intrinsic Matrix from CamParafile 
        K,_,_ = readCamParaFile("cam_para/MOT17/"+seq+".txt")
        
        if K.shape[1] >3:
            K = K[:, :3] # Make sure 3x3
        K = K.astype(np.float32)
        f = (K[0][0], K[1][1])
        c = (K[0][2], K[1][2])
        
        # src_img = f"dataset/MOT17/train/{seq}/img1/000001.jpg"
        # det_file = f"det_results/mot17/yolox_x_ablation/{seq}.txt"
        # test_distort_new(src_img, f, c, dist_coeff, det_file, cam_type)
        
        # Create distorted detection result 
        src_det = f"det_results/mot17/yolox_x_ablation/{seq}.txt"
        dst_det = f"det_results/mot17_dist/yolox_x_ablation/{seq}.txt"
        dst_det_dataset = f"dataset/MOT17_dist/train/{seq}/det/det.txt"
        create_distorted_det_results(src_det, dst_det, dist_coeff, cam_type)
        create_distorted_det_results(src_det, dst_det_dataset, dist_coeff, cam_type)

        # seqinforfile 
        src_seqinfo_file = f"dataset/MOT17/train/{seq}/seqinfo.ini"
        dst_seqinfo_folder = f"dataset/MOT17_dist/train/{seq}/"
        copy_file2dir(src_seqinfo_file, dst_seqinfo_folder)
        
        # Create cammera parameter file
        src_cam_para = f"cam_para/MOT17/{seq}.txt"
        dst_cam_para = f"cam_para/MOT17_dist/{seq}.txt"
        create_cam_para(src_cam_para, dst_cam_para, dist_coeff)
        
        # Create distorted gt 
        src_mot = f"dataset/MOT17/train/{seq}/gt/gt.txt"
        dst_mot = f"dataset/MOT17_dist/train/{seq}/gt/gt.txt"
        create_distorted_mot17(src_mot, dst_mot, dist_coeff, cam_type)
