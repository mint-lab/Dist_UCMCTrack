import shutil
import os


if __name__ == '__main__':
    sub_dir = 'MOT20/images/test'
    output_dir = 'output/mot20'
    exp_name = 'test'
    mot_path = 'dataset'
    seq_nums = ['MOT20-04',
                'MOT20-06',
                'MOT20-07',
                'MOT20-08']

    predict_path = os.path.join(output_dir, exp_name)
    sub_dir = 'MOT20/train'
    seq_nums = os.listdir('dataset/MOT20/train')
    accs = []
    seqs = []
    for seq_num in seq_nums:
        shutil.copyfile(os.path.join(mot_path, sub_dir, f'{seq_num}/gt/gt.txt'),os.path.join(predict_path,f'{seq_num}.txt'))