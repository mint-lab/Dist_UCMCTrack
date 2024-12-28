import numpy as np
import json, pickle, copy
from scipy.spatial.transform import Rotation
from pyproj import Transformer
import matplotlib.colors as mcolors
import cv2 as cv
import os 

def save_camera_config(json_file, cameras, keys_to_save=['name', 'focal', 'center', 'distort', 'rvec', 'tvec']):
    '''Save the multi-camera configuration as a JSON file'''
    with open(json_file, 'w') as f:
        cameras_to_save = []
        for cam in cameras:
            cam_to_save = {}
            for key in keys_to_save:
                if key in cam:
                    if type(cam[key]) == np.ndarray:
                        cam_to_save[key] = cam[key].tolist()
                    else:
                        cam_to_save[key] = cam[key]
            cameras_to_save.append(cam_to_save)
        json.dump(cameras_to_save, f, indent=4)

def postprocess_camera_config(cameras):
    '''Post-process the multi-camera configuration'''
    for cam in cameras:
        for key in ['K', 'distort', 'rvec', 'tvec', 'ori', 'pos']:
            if key in cam:
                cam[key] = np.array(cam[key])
        if ('focal' in cam) and ('center' in cam):
            cam['K'] = np.array([[cam['focal'][0], 0, cam['center'][0]], [0, cam['focal'][1], cam['center'][1]], [0, 0, 1]])
        if ('rvec' in cam) and ('tvec' in cam):
            cam['ori'] = Rotation.from_rotvec(cam['rvec']).as_matrix().T
            cam['pos'] = -cam['ori'] @ cam['tvec']
        if 'polygons' in cam:
            cam['polygons'] = {int(key): np.array(value).reshape(-1, 2) for key, value in cam['polygons'].items()}
        else:
            cam['polygons'] = {}
        if 'cylinder_file' in cam:
            with open(cam['cylinder_file'], 'rb') as f:
                cam['cylinder_table'] = pickle.load(f)
        if 'cuboid_file' in cam:
            with open(cam['cuboid_file'], 'rb') as f:
                cam['cuboid_table'] = pickle.load(f)

def load_camera_config(json_file, cameras=None):
    '''Load the multi-camera configuration from a JSON file'''
    with open(json_file, 'r') as f:
        cameras_from_file = json.load(f)
        if cameras is None:
            cameras = cameras_from_file
        else:
            for (src, dst) in zip(cameras, cameras_from_file):
                src.update(dst)
        postprocess_camera_config(cameras)
        return cameras

def postprocess_satellite_config(satellite):
    '''Post-process the satellite configuration'''
    for key in ['pts', 'planes']:
        if key in satellite:
            satellite[key] = np.array(satellite[key])
    if 'planes' not in satellite:
        satellite['planes'] = []
    if 'roads' in satellite:
        satellite['roads'] = [np.array(road).reshape(-1, 2) for road in satellite['roads']]
        roads_data = []
        for road in satellite['roads']:
            road_m = np.array([conv_pixel2meter(pt, satellite['origin_pixel'], satellite['meter_per_pixel']) for pt in road])
            road_v = road_m[1:] - road_m[:-1]
            road_n = np.linalg.norm(road_v, axis=1)
            roads_data.append(np.hstack((road_m[:-1], road_v, road_n.reshape(-1, 1))))
        satellite['roads_data'] = np.vstack(roads_data)
    else:
        satellite['roads'] = []
        satellite['roads_data'] = []

def load_satellite_config(json_file, satellite=None):
    '''Load the satellite configuration from a JSON file'''
    with open(json_file, 'r') as f:
        satellite_from_file = json.load(f)
        if satellite is None:
            satellite = satellite_from_file
        else:
            satellite.update(satellite_from_file)

        postprocess_satellite_config(satellite)
        return satellite

def get_default_config():
    config = {
        'detector_name'     : 'YOLOv5',
        'detector_option'   : {},
        'tracker_name'      : 'DeepSORT',
        'tracker_option'    : {},
        'tracker_margin'    : 1.2,
        'filter_classes'    : [0, 2],
        'filter_min_conf'   : 0.5,
        'filter_rois'       : [],
        'filter_max_dist'   : 50.0,
        'multicam_name'     : 'Simple',
        'multicam_option'   : {},
        'zoom_level'        : 1.0,
        'frame_offset'      : (10, 10),
        'frame_color'       : (0, 255, 0),
        'frame_font_scale'  : 0.7,
        'label_offset'      : (-8, -24),
        'label_font_scale'  : 0.5,
        'circle_radius'     : 8,
        'bbox_thickness'    : 3,
        'bbox_skip_color'   : (127, 127, 127)
    }
    return config

def postprocess_config(config):
    if 'filter_rois' in config:
        if (len(config['filter_rois']) > 0) and (type(config['filter_rois']) is not dict):
            config['filter_rois'] = {idx: np.array(polygon).astype(np.float32).reshape(-1, 2) for idx, polygon in enumerate(config['filter_rois'])}

def load_config(json_file):
    '''Load the satellite and multi-camera configuration together from a JSON file'''
    with open(json_file, 'r') as f:
        config = json.load(f)
        if ('satellite' in config) and ('cameras' in config) and ('config' in config):
            postprocess_satellite_config(config['satellite'])
            postprocess_camera_config(config['cameras'])
            default_cfg = get_default_config()
            default_cfg.update(config['config'])
            config['config'] = default_cfg
            postprocess_config(config['config'])
            for cam in config['cameras']:
                # Copy empty options from the global options
                cam_cfg = copy.deepcopy(config['config'])
                cam_cfg.update(cam['config'])
                cam['config'] = cam_cfg
                postprocess_config(cam['config'])
            return config['satellite'], config['cameras'], config['config']
    return {}, [], {}

def load_3d_points(csv_file, trans_code='', origin_id=-1):
    '''Load 3D points (e.g. road markers) from a CSV file'''
    # Read the CSV file
    id_pts = np.loadtxt(csv_file, delimiter=',')
    pts = {int(id): np.array(pt) for id, *pt in id_pts}

    # Transform the given data to the specific coordinate
    if trans_code:
        transformer = Transformer.from_crs('EPSG:4326', trans_code)
        for id, (lon, lat, alt) in pts.items():
            y, x = transformer.transform(lat, lon)
            pts[id] = np.array([x, y, float(alt)])

    # Assign the origin using the given index
    if origin_id >= 0:
        origin = pts[origin_id]
        for id, pt in pts.items():
            pts[id] = pt - origin
    return pts

def conv_pixel2meter(pt, origin_pixel, meter_per_pixel):
    '''Convert image position to metric position on the satellite image'''
    x = (pt[0] - origin_pixel[0]) * meter_per_pixel
    y = (origin_pixel[1] - pt[1]) * meter_per_pixel
    z = 0
    if len(pt) > 2:
        z = pt[2]
    if type(pt) is np.ndarray:
        return np.array([x, y, z])
    return [x, y, z]

def conv_meter2pixel(pt, origin_pixel, meter_per_pixel):
    '''Convert metric position to image position on the satellite image'''
    u = pt[0] / meter_per_pixel + origin_pixel[0]
    v = origin_pixel[1] - pt[1] / meter_per_pixel
    if type(pt) is np.ndarray:
        return np.array([u, v])
    return [u, v]

def load_3d_points_from_satellite(json_file, origin_id=-1):
    '''Load 3D points (e.g. road markers) from 2D points defined on the satellite image'''
    satellite = load_satellite_config(json_file)
    if ('id_pts' in satellite) and ('meter_per_pixel' in satellite):
        # Copy points from the given 'satellite'
        pts = {}
        for id, u, v in satellite['id_pts']:
            pts[int(id)] = satellite['meter_per_pixel'] * np.array([u, -v, 0])

        # Assign the origin using the given index
        if origin_id >= 0:
            origin = pts[origin_id]
            for id, pt in pts.items():
                pts[id] = pt - origin
        return pts

def get_marker_palette(int_type=False, bgr=False):
    '''Load the pre-defined palette for consistent coloring'''
    # Use 'TABLEAU_COLORS' palette by default
    palette = [mcolors.ColorConverter.to_rgb(rgb) for rgb in mcolors.TABLEAU_COLORS.values()]
    palette[7] = (0., 0., 0.) # Make gray to black for better visibility
    if int_type:
        palette = [(int(255* r), int(255* g), int(255* b)) for r, g, b in palette]
    if bgr:
        palette = [(b, g, r) for r, g, b in palette]
    return palette

def cam_para_to_config(cam_para_file, save_file):
    from detector.mapper import readCamParaFile
    
    K, R, T, flag = readCamParaFile(cam_para_file, flag_KRT=True)
    rvec, _ = cv.Rodrigues(R)
    cam_name = cam_para_file.split("/")[-1][:-4]
    if not os.path.exists(save_file):
        config = {"satellite": {"planes": [[0,0,1,0]]}, "cameras": [{"name": cam_name,
            "focal": [
                0.0,
                0.0
            ],
            "center": [
                959.0,
                539.0
            ],
            "distort": [
                0.5163732841633568,
                0.0,
                0.0,
                0.0,
                0.0
            ],
            "rvec": [
                1.937769333339697,
                -0.007887800551032147,
                -0.005418428934453571
            ],
            "tvec": [
                0.0,
                0.011,
                0.004
            ],
            "config": {},
            "polygons": {}
        }
    ],
    "config": {}}
        with open(save_file, 'w') as f:
            json.dump(config, f)

    with open(save_file, 'r') as f:
        config = json.load(f)
        camera_config = config["cameras"][0]

        camera_config["focal"] = [K[0,0], K[1,1]]
        camera_config["center"] = [K[0,2], K[1,2]]
        camera_config["rvec"] = rvec.flatten().tolist()
        camera_config["tvec"] = T.tolist()
            
    with open(save_file, 'w') as f:
        json.dump(config, f, indent=4)  

def trim_object(obj, image_size):
    '''Trim 2D points within the given image size'''
    image_w, image_h = image_size
    obj[(obj[:,0] < 0), 0] = 0
    obj[(obj[:,1] < 0), 1] = 0
    obj[(obj[:,0] >= image_w), 0] = image_w - 1
    obj[(obj[:,1] >= image_h), 1] = image_h - 1

def get_object_bottom_mid(obj):
    '''Get the bottom middle point of the given 2D points'''
    return [(min(obj[:,0]) + max(obj[:,0])) / 2, max(obj[:,1])]

def gen_cylinder_data(satellite, camera, image_size, image_step, cylinder_shape=(0.3, 1.6)):
    '''Generate a lookup table for cylinders for the specific camera'''
    image_w, image_h = image_size
    data = []
    for y in range(0, image_h, image_step):
        for x in range(0, image_w, image_step):
            center = np.array((x, y))
            obj, _ = get_cylinder(center, *cylinder_shape, satellite, camera)
            if obj is not None:
                trim_object(obj, image_size)
                bottom_mid = get_object_bottom_mid(obj)
                delta = center - bottom_mid
                data.append(bottom_mid + delta.tolist())
    return np.array(data)

def save_lookup_table(config_file, image_size=(1920, 1080), image_step=100, save_prefix=''):
    '''Generate and save lookup tables (for cylinders) for multiple cameras to pickle files'''
    satellite, cameras, _ = load_config(config_file)
    for idx, cam in enumerate(cameras):
        data = gen_cylinder_data(satellite, cam, image_size, image_step)
        with open(save_prefix + cam['name'] + '_cylinder.pickle', 'wb') as f:
            pickle.dump(data, f)

def predict_center_from_table(bottom_mid, table, dist_threshold=100):
    '''Predict a foot point using the given lookup table and nearest search'''
    x, y = bottom_mid
    dist = np.fabs(table[:,0] - x) + np.fabs(table[:,1] - y)
    min_idx = np.argmin(dist)
    if dist[min_idx] < dist_threshold:
        return table[min_idx,2:4]
    return np.zeros(2)

def draw_bbox(image, obj_p, color, thickness=2):
    '''Draw a bounding box of the given 2D points'''
    tl = np.array((min(obj_p[:,0]), min(obj_p[:,1])))
    br = np.array((max(obj_p[:,0]), max(obj_p[:,1])))
    cv.rectangle(image, tl.astype(np.int32), br.astype(np.int32), color, thickness)
    return tl, br

def test_table(image_file, config_file, camera_name='camera', cylinder_shape=(0.3, 1.6), cuboid_shape=(1.8, 4.5, 1.4)):
    '''Test a lookup table which predicts a foot point'''

    # A callback function to save the clicked point
    def click_camera_image(event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            param[0] = x
            param[1] = y

    # Configure parameters
    object_color = (100, 100, 100)
    cursor_radius = 10
    cursor_color = (0, 0, 255)
    bbox_color = (255, 0, 0)
    predict_radius = 5

    # Load configuration and images
    satellite, cameras, _ = load_config(config_file)
    cam = next(filter(lambda cam: cam['name'] == camera_name, cameras))
    
    camview = cv.imread(image_file)

    camview_size = camview.shape[0:2][::-1]

    # Get a point and draw an object at the point
    cylinder_mode = True
    click_curr, click_prev = np.array([0, 0]), np.array([0, 0])
    cv.imshow('test_table', camview)
    cv.setMouseCallback('test_table', click_camera_image, click_curr)
    while True:
        if not np.array_equal(click_curr, click_prev):
            click_prev = click_curr.copy()

            # Show the point and draw an object at the point
            camview_viz = camview.copy()
            bottom_mid, delta = click_curr, np.zeros(2)
            if cylinder_mode:
                # Draw a cylinder on the point
                cylinder, _ = get_cylinder(click_curr, *cylinder_shape, satellite, cam)
                if cylinder is not None:
                    trim_object(cylinder, camview_size)
                    bottom_mid = get_object_bottom_mid(cylinder)
                    if 'cylinder_table' in cam:
                        delta = predict_center_from_table(bottom_mid, cam['cylinder_table'])
                    draw_cylinder(camview_viz, cylinder, object_color)
                    draw_bbox(camview_viz, cylinder, bbox_color)
            else:
                # Draw a cuboid on the point
                direction = get_road_direction(click_curr, satellite, cam)
                cuboid, _ = get_cuboid(click_curr, *cuboid_shape, direction, satellite, cam)
                if cuboid is not None:
                    trim_object(cuboid, camview_size)
                    bottom_mid = get_object_bottom_mid(cuboid)
                    if 'cuboid_table' in cam:
                        delta = predict_center_from_table(bottom_mid, cam['cuboid_table'])
                    draw_cuboid(camview_viz, cuboid, object_color)
                    draw_bbox(camview_viz, cuboid, bbox_color)

            # Draw 'click_curr' as a cross mark and the predicted center as a circle
            cv.line(camview_viz, click_curr-[cursor_radius, 0], click_curr+[cursor_radius, 0], cursor_color, 2)
            cv.line(camview_viz, click_curr-[0, cursor_radius], click_curr+[0, cursor_radius], cursor_color, 2)
            center = bottom_mid + delta
            cv.circle(camview_viz, center.astype(np.int32), predict_radius, bbox_color, -1)

            cv.imshow('test_table', camview_viz)

        key = cv.waitKey(1)
        if key == ord('\t'): # Tab
            cylinder_mode = not cylinder_mode
        elif key == 27:      # ESC
            break

    cv.destroyAllWindows()
def put_object_on_plane(center_p, object_m, direction, satellite, camera):
    '''Put 3D points (unit: [meter]) on the given point (unit: [pixel]) and direction (unit: [meter])'''
    center_m, _ = localize_point(center_p, camera['K'], camera['distort'], camera['ori'], camera['pos'], camera['polygons'], satellite['planes'])
    if center_m is not None:
        rz = np.array([0, 0, 1])
        plane_idx = check_polygons(center_p, camera['polygons'])
        if (plane_idx >= 0) and (plane_idx < len(satellite['planes'])):
            rz = satellite['planes'][plane_idx][0:3]
            rz = rz / np.linalg.norm(rz)
        rx = np.array([direction[0], direction[1], -(rz[0:2].T @ direction[0:2]) / rz[-1]])
        rx = rx / np.linalg.norm(rx)
        ry = np.cross(rz, rx)
        R = np.vstack((rx, ry, rz)).T
        return center_m + object_m @ R.T
    return None

def get_circle(center_p, radius_m, satellite, camera, offset_m=0, n=16):
    '''Generate a pair of points (unit: [pixel] and [meter]) for a circle on the given point (unit: [pixel])'''
    thetas = np.arange(0, 2 * np.pi, 2 * np.pi / n)
    circle = np.array([[radius_m * np.cos(theta), radius_m * np.sin(theta), offset_m] for theta in thetas])
    circle_m = put_object_on_plane(center_p, circle, (1, 0), satellite, camera)
    if circle_m is None:
        return None, None
    circle_p, _ = cv.projectPoints(circle_m, camera['rvec'], camera['tvec'], camera['K'], camera['distort'])
    return circle_p.reshape(-1, 2), circle_m

def get_cylinder(center_p, radius_m, height_m, satellite, camera, offset_m=0, n=32):
    '''Generate a pair of points (unit: [pixel] and unit: [meter]) for a cylinder on the given point (unit: [pixel])'''
    bot_p, bot_m = get_circle(center_p, radius_m, satellite, camera, offset_m, n)
    top_p, top_m = get_circle(center_p, radius_m, satellite, camera, offset_m + height_m, n)
    if (bot_p is None) or (top_p is None):
        return None, None
    return np.vstack((bot_p, top_p)), np.vstack((bot_m, top_m))

def draw_cylinder(image, cylinder_p, color, thickness=2):
    '''Draw a cylinder described as 2D points (unit: [pixel])'''
    half = int(len(cylinder_p)/2)
    bottom, top = cylinder_p[:half], cylinder_p[half:]
    bl_idx, tl_idx = np.argmin(bottom[:,0]), np.argmin(top[:,0])
    br_idx, tr_idx = np.argmax(bottom[:,0]), np.argmax(top[:,0])
    bottom, top = bottom.astype(np.int32), top.astype(np.int32)
    cv.polylines(image, [bottom, top], True, color, thickness)
    cv.line(image, bottom[bl_idx], top[tl_idx], color, thickness)
    cv.line(image, bottom[br_idx], top[tr_idx], color, thickness)

def get_rectangle(center_p, front_m, side_m, direction, satellite, camera, offset_m=0):
    '''Generate a pair of points (unit: [pixel] and [meter]) for a rectangle on the given point (unit: [pixel]) and direction (unit: [meter])'''
    f_half, s_half = front_m / 2, side_m / 2
    rect = np.array([[-s_half, f_half, offset_m], [s_half, f_half, offset_m], [s_half, -f_half, offset_m], [-s_half, -f_half, offset_m]])
    rect_m = put_object_on_plane(center_p, rect, direction, satellite, camera)
    if rect_m is None:
        return None, None
    rect_p, _ = cv.projectPoints(rect_m, camera['rvec'], camera['tvec'], camera['K'], camera['distort'])
    return rect_p.reshape(-1, 2), rect_m

def localize_point(pt, K, distort=None, ori=np.eye(3), pos=np.zeros((3, 1)), polygons={}, planes=[]):
    '''Calculate 3D location (unit: [meter]) of the given point (unit: [pixel]) with the given camera configuration'''
    # Make a ray aligned to the world coordinate
    pt = np.array(pt, dtype=np.float32)
    pt_n = cv.undistortPoints(np.array(pt, dtype=K.dtype), K, distort).flatten()
    r = ori @ np.append(pt_n, 1) # A ray with respect to the world coordinate
    scale = np.linalg.norm(r)
    r = r / scale

    # Get a plane if 'pt' exists inside of any 'polygons'
    n, d = np.array([0, 0, 1]), 0
    plane_idx = check_polygons(pt, polygons)
    if (plane_idx >= 0) and (plane_idx < len(planes)):
        n, d = planes[plane_idx][0:3], planes[plane_idx][-1]

    # Calculate distance and position on the plane
    denom = n.T @ r
    if np.fabs(denom) < 1e-6: # If the ray 'r' is almost orthogonal to the plane norm 'n' (~ almost parallel to the plane)
        return None, None
    distance = -(n.T @ pos + d) / denom
    r_c = ori.T @ (np.sign(distance) * r)
    if r_c[-1] <= 0: # If the ray 'r' stretches in the negative direction (negative Z)
        return None, None
    position = pos + distance * r
    return position, np.fabs(distance)

def check_polygons(pt, polygons):
    '''Check whether the given point belongs to polygons (index) or not (-1)'''
    if len(polygons) > 0:
        for idx, polygon in polygons.items():
            if cv.pointPolygonTest(polygon, np.array(pt, dtype=np.float32), False) >= 0:
                return idx
    return -1

def get_uncertainty(pt, sigma, K, distort=None, ori=np.eye(3), pos=np.zeros((3, 1)), polygons={}, planes=[], n=32):
    '''Get an uncertainty ellipse on the given point (unit: [pixel]) with the standard deviation (unit: [pixel])'''
    thetas = np.arange(0, 2 * np.pi, 2 * np.pi / n)
    circle_p = pt + sigma * np.array([[np.cos(theta), np.sin(theta)] for theta in thetas])
    circle_m = [localize_point(p, K, distort, ori, pos, polygons, planes)[0] for p in circle_p]
    return circle_p, np.array(circle_m)

def get_bbox_bottom_mid(bbox):
    '''Get the bottom middle point of the given bounding box'''
    tl_x, tl_y, br_x, br_y = bbox
    return np.array([(tl_x + br_x) / 2, br_y])

def get_road_direction(pt_p, satellite, camera, offset_m=0, dist_threshold=10):
    '''Find the nearest road direction of the given point (unit: [pixel]) from satellite['roads_data']'''
    pt_m, _ = localize_point(pt_p, camera['K'], camera['distort'], camera['ori'], camera['pos'], camera['polygons'], satellite['planes'])
    if pt_m is not None:
        p = pt_m[:2]
        nearest_dist = dist_threshold
        nearest_idx = -1
        for idx, data in enumerate(satellite['roads_data']):
            p0, v, n = data[:2], data[3:5], data[-1]
            delta = p - p0
            proj_ratio = (delta @ v) / n
            if proj_ratio < 0:
                proj_p = p0
            elif proj_ratio > 1:
                proj_p = p0 + v
            else:
                proj_p = p0 + proj_ratio * v
            dist = np.linalg.norm(proj_p - p)
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_idx = idx
        if nearest_idx >= 0:
            return satellite['roads_data'][nearest_idx, 3:5]
    return np.array([1, 0])

def get_cuboid(center_p, front_m, side_m, height_m, direction, satellite, camera, offset_m = 0):
    '''Generate a pair of points (unit: [pixel] and [meter]) for a cuboid on the given point (unit: [pixel]) and direction (unit: [meter])'''
    bot_p, bot_m = get_rectangle(center_p, front_m, side_m, direction, satellite, camera, offset_m)
    top_p, top_m = get_rectangle(center_p, front_m, side_m, direction, satellite, camera, offset_m + height_m)
    if (bot_p is None) or (top_p is None):
        return None, None
    return np.vstack((bot_p, top_p)), np.vstack((bot_m, top_m))

def draw_cuboid(image, cuboid_p, color, thickness=2):
    '''Draw a cuboid described as 2D points (unit: [pixel])'''
    half = int(len(cuboid_p)/2)
    cuboid = cuboid_p.astype(np.int32)
    bottom, top = cuboid[:half], cuboid[half:]
    cv.polylines(image, [bottom, top], True, color, thickness)
    for b, t in zip(bottom, top):
        cv.line(image, b, t, color, thickness)
if __name__ == '__main__':
    # Test 'load_3d_points()'
    # markers3d = load_3d_points('data/ETRITestbed/markers45_QGIS+MMS.csv', trans_code='EPSG:5186', origin_id=23)

    # Test 'load_3d_points_satellit()'
    # markers3d_sate = load_3d_points_from_satellite('data/ETRITestbed/markers45_satellite.json', origin_id=23)

    # Test cam_para_to_config 
    # cam_para_to_config(cam_para_file="cam_para/MOT17/MOT17-02-SDP.txt", save_file="detector/config_mot17_02.json")
    # cam_para_to_config(cam_para_file="cam_para/MOT17/MOT17-04-SDP.txt", save_file="detector/config_mot17_04.json")
    cam_para_to_config(cam_para_file="cam_para/MOT17/MOT17-05-SDP.txt", save_file="detector/config_mot17_05.json")    
    # cam_para_to_config(cam_para_file="cam_para/MOT17/MOT17-09-SDP.txt", save_file="detector/config_mot17_09.json")
    cam_para_to_config(cam_para_file="cam_para/MOT17/MOT17-10-SDP.txt", save_file="detector/config_mot17_10.json")
    cam_para_to_config(cam_para_file="cam_para/MOT17/MOT17-11-SDP.txt", save_file="detector/config_mot17_11.json")
    cam_para_to_config(cam_para_file="cam_para/MOT17/MOT17-13-SDP.txt", save_file="detector/config_mot17_13.json") 
   
    # save_lookup_table('detector/config_mot17_02.json', image_step=10, save_prefix = "detector/data/")
    save_lookup_table('detector/config_mot17_04.json', image_step=10, save_prefix = "detector/data/")
    # save_lookup_table('detector/config_mot17_05.json', image_step=10, save_prefix = "detector/data/")
    # save_lookup_table('detector/config_mot17_09.json', image_step=10, save_prefix = "detector/data/")
    # save_lookup_table('detector/config_mot17_10.json', image_step=10, save_prefix = "detector/data/")
    # save_lookup_table('detector/config_mot17_11.json', image_step=10, save_prefix = "detector/data/")
    # save_lookup_table('detector/config_mot17_13.json', image_step=10, save_prefix = "detector/data/")


    # test_table('detector/data/MOT17_02_screenshot.png', 'detector/config_mot17_02.json', camera_name='MOT17_02')
    test_table('detector/data/MOT17_04_screenshot.png', 'detector/config_mot17_04.json', camera_name='MOT17_04')
    # test_table('detector/data/MOT17_09_screenshot.png', 'detector/config_mot17_09.json', camera_name='MOT17_09')

    # distort lookup table 
 