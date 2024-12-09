from util.run_ucmc import run_ucmc, make_args

if __name__ == '__main__':

    det_path = "det_results/mot20_dist"
    cam_path = "cam_para/MOT20_dist"
    gmc_path = "gmc/mot20"
    out_path = "output/mot20_dist"
    exp_name = "val_ori"
    dataset = "MOT20_dist" 
    args = make_args()
    print(args)

    run_ucmc(args, det_path, cam_path, gmc_path, out_path, exp_name,dataset)
