import cv2
from tracker.ucmc import UCMCTrack
from detector.detector import Detector
from detector.mapper import Mapper
# from test import get_cylinder, draw_cylinder, localize_point, load_config
import numpy as np
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Ellipse
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

"""
Test code using top-view visualization
"""
class Detection:
    def __init__(self, id, bb_left=0, bb_top=0, bb_width=0, bb_height=0, conf=0, det_class=0):
        self.id = id
        self.bb_left = bb_left
        self.bb_top = bb_top
        self.bb_width = bb_width
        self.bb_height = bb_height
        self.conf = conf
        self.det_class = det_class
        self.track_id = 0
        self.y = np.zeros((2, 1))
        self.R = np.eye(4)

    def __str__(self):
        return 'd{}, bb_box:[{},{},{},{}], conf={:.2f}, class{}, uv:[{:.0f},{:.0f}], mapped to:[{:.1f},{:.1f}]'.format(
            self.id, self.bb_left, self.bb_top, self.bb_width, self.bb_height, self.conf, self.det_class,
            self.bb_left+self.bb_width/2, self.bb_top+self.bb_height, self.y[0,0], self.y[1,0])

    def __repr__(self):
        return self.__str__()

class DetectorDemo:
    def __init__(self):
        self.seq_length = 0
        self.gmc = None
        self.det_res = None  

    def load(self, cam_para_file):
        self.mapper = Mapper(cam_para_file, "MOT17")
    
        det_results = 'det_results/'
        _, dataset_name, seq_name = cam_para_file.split("/")
        if 'mot17' in dataset_name.lower():
            det_results = det_results + 'mot17/bytetrack_x_mot17/' + seq_name
        else:
            det_results = det_results + 'mot20/' + seq_name 

        with open(det_results, 'r') as f:
            self.det_res = f.readlines()

    def get_dets(self, img, conf_thresh=0, det_classes=[0]):
        dets = []

        # Convert frame from BGR to RGB (because OpenCV uses BGR format)
        frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
    
        det_id = 0
        cls_id = 0
        for results in self.det_res:
            frame_seq, id, bb_left, bb_top, w, h, conf, x, y, z = results.split(',')
            w = float(w)
            h = float(h)
            conf = float(conf)
            if w <= 10 and h <= 10 or cls_id not in det_classes or conf <= conf_thresh:
                continue

            # Create a new Detection object
            det = Detection(det_id)
            det.bb_left = float(bb_left)
            det.bb_top = float(bb_top)
            det.bb_width = float(w)
            det.bb_height = float(h)
            det.conf = float(conf)
            det.det_class = int(cls_id)
            det.y, det.R = self.mapper.mapto([det.bb_left, det.bb_top, det.bb_width, det.bb_height])
            det_id += 1
            dets.append(det)

        return dets

def top_view_multi_model(args, video, det, gmc):
    plt.rcParams['figure.max_open_warning'] = 0
    class_dict = {"person": 0}
    cap = cv2.VideoCapture(video)

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_out = cv2.VideoWriter("top_view.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    cv2.namedWindow("demo", cv2.WINDOW_NORMAL)

    # Detectors initialization
    d_o = Detector()  # original calibration
    d_o.load("cam_para/MOT17/MOT17-04-SDP.txt", det_file=det, gmc_file=gmc)

    d_n = Detector()  # new calibration
    d_n.load("cam_para/MOT17_ped_calib/MOT17-04-SDP.txt", det_file=det, gmc_file=gmc)

    t_o = UCMCTrack(args.a, args.a, args.wx, args.wy, args.vmax, args.cdt, fps, "MOT", args.high_score, False, d_o)
    t_n = UCMCTrack(args.a, args.a, args.wx, args.wy, args.vmax, args.cdt, fps, "MOT", args.high_score, False, d_n)

    frame_id = 1

    while True:
        # Initialize Matplotlib figure and axis
        dpi = 100
        fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
        ax = fig.add_subplot(111)
        ax.set_xlim(-100, 100)
        ax.set_ylim(-100, 100)
        ax.set_aspect('equal')
        ax.set_facecolor('white')  # Set background color to white

        # Get detections and update trackers
        ds_o = d_o.get_dets(frame_id, args.conf_thresh, class_dict['person'])
        t_o.update(ds_o, frame_id)

        ds_n = d_n.get_dets(frame_id, args.conf_thresh, class_dict['person'])
        t_n.update(ds_n, frame_id)

        # Plot detections
        for det_o, det_n in zip(ds_o, ds_n):

            if det_o.track_id > 0:
                x, y = det_o.y[0, 0], det_o.y[1, 0]
                ax.plot(x, y, 'ro')
                ax.text(x, y, f'ID: {det_o.track_id}', fontsize=12, color='red')

                eigvals, eigvecs = np.linalg.eig(det_o.R[:2, :2])
                order = eigvals.argsort()[::-1]
                eigvals, eigvecs = eigvals[order], eigvecs[:, order]
                angle = np.arctan2(*eigvecs[:, 0][::-1]) * 180 / np.pi
                ell_width, ell_height = 2 * np.sqrt(eigvals)
                ellipse = Ellipse((x, y), ell_width, ell_height, angle=angle, edgecolor='blue', facecolor='none')
                ax.add_patch(ellipse)

            if det_n.track_id > 0:
                x, y = det_n.y[0, 0], det_n.y[1, 0]
                ax.plot(x, y, 'go')
                ax.text(x, y, f'ID: {det_n.track_id}', fontsize=12, color='green')

                eigvals, eigvecs = np.linalg.eig(det_n.R[:2, :2])
                order = eigvals.argsort()[::-1]
                eigvals, eigvecs = eigvals[order], eigvecs[:, order]
                angle = np.arctan2(*eigvecs[:, 0][::-1]) * 180 / np.pi
                ell_width, ell_height = 2 * np.sqrt(eigvals)
                ellipse = Ellipse((x, y), ell_width, ell_height, angle=angle, edgecolor='green', facecolor='none')
                ax.add_patch(ellipse)

        # Convert Matplotlib figure to OpenCV image
        canvas = FigureCanvas(fig)
        canvas.draw()
        frame_img = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
        frame_img = frame_img.reshape((int(height), int(width), 3))

        # Display and write video
        cv2.imshow("demo", frame_img)
        video_out.write(frame_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_id += 1

def viz_traj(args, seq, config_file):
    plt.rcParams['figure.max_open_warning'] = 0
    class_dict = {"person": 0}
    cap = cv2.VideoCapture(args.video)

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_out = cv2.VideoWriter(args.output_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))  

    # Open a cv2 window with specified height and width
    cv2.namedWindow("demo", cv2.WINDOW_NORMAL)

    if args.switch_2D:
        cv2.resizeWindow("demo", width, height)

    detector_mapper = Detector(flag_unpro=False, lookup_table=args.lookup_table)
    detector_mapper.load(args.cam_para, det_file=args.det_result, gmc_file=args.gmc, switch_2D=False)
    tracker_m = UCMCTrack(args.a, args.a, args.wx, args.wy, args.vmax, args.cdt, fps, "MOT", args.high_score, False, detector_mapper)

    frame_id = 1
    specific_track_id = 22 # 특정 트랙 ID를 추적하기 위한 파라미터
    trajectory_m = []  # 특정 트랙 ID의 위치를 저장할 리스트
    trajectory_u = []

    while True:
        
        # Initialize Matplotlib figure and axis
        dpi = 50  # Explicit DPI setting
        fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
        ax = fig.add_subplot(111)
        ax.set_xlim(-10, 20)  # Set limits for x-axis
        ax.set_ylim(-10, 20)  # Set limits for y-axis
        ax.set_aspect('equal')

        frame_img = np.ones((height, width, 3), dtype=np.uint8) * 255  # White background
        dets_m = detector_mapper.get_dets(frame_id, args.conf_thresh, class_dict['person'])
        tracker_m.update(dets_m, frame_id)

        for det_m in dets_m:
            if det_m.track_id > 0:
                print("track_id in M:", det_m.track_id)
                x_m, y_m = det_m.y[0, 0], det_m.y[1, 0]
                if det_m.track_id == specific_track_id:
                    trajectory_m.append((x_m, y_m))
                
        # 특정 트랙 ID의 전체 경로 시각화
        if len(trajectory_m) > 1:
            trajectory_np_m = np.array(trajectory_m)
            ax.plot(trajectory_np_m[:, 0], trajectory_np_m[:, 1], 'g-', linewidth=2)  # 빨간 선으로 경로 시각화

        # 특정 트랙 ID의 전체 경로 시각화
        if len(trajectory_u) > 1:
            trajectory_np_u = np.array(trajectory_u)
            ax.plot(trajectory_np_u[:, 0], trajectory_np_u[:, 1], 'b-', linewidth=5)  # 파란색 선으로 경로 시각화

        # Convert Matplotlib figure to image
        canvas = FigureCanvas(fig)
        canvas.draw()
        frame_img = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
        
        # Ensure the correct reshaping by adjusting dimensions
        frame_img = frame_img.reshape((int(height), int(width), 3))
        
        cv2.imshow("demo", frame_img)
        video_out.write(frame_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
        frame_id +=1

if __name__ == "__main__":

    from util.run_ucmc import  make_args

    det_path = "det_results/mot17/yolox_x_ablation/MOT17-04-SDP.txt"
    vid_name = "MOT17_04.avi"
    cam_path = "cam_para/MOT17"
    gmc_path = "gmc/mot17/GMC-MOT17-04.txt"
    out_path = "output/mot17"
    exp_name = "val"
    dataset = "MOT17"
    args = make_args()

    top_view_multi_model(args, vid_name, det_path, gmc_path)
