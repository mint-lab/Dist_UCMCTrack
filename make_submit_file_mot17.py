import shutil
import os


if __name__ == '__main__':
    sub_dir = 'MOT17/images/test'
    output_dir = 'output/mot17'
    exp_name = 'test'
    mot_path = 'dataset'
    seq_nums = ['MOT17-01-SDP',
                'MOT17-03-SDP',
                'MOT17-06-SDP',
                'MOT17-07-SDP',
                'MOT17-08-SDP',
                'MOT17-12-SDP',
                'MOT17-14-SDP']


    """copy reuslts for same sequences"""
    repeated_seq_nums = ['MOT17-01-DPM',
                        'MOT17-03-DPM',
                        'MOT17-06-DPM',
                        'MOT17-07-DPM',
                        'MOT17-08-DPM',
                        'MOT17-12-DPM',
                        'MOT17-14-DPM',
                        'MOT17-01-FRCNN',
                        'MOT17-03-FRCNN',
                        'MOT17-06-FRCNN',
                        'MOT17-07-FRCNN',
                        'MOT17-08-FRCNN',
                        'MOT17-12-FRCNN',
                        'MOT17-14-FRCNN']
    
    print('copy reuslts for same sequences: ')
    predict_path = os.path.join(output_dir, exp_name)
    for repeated_seq_nums_i in repeated_seq_nums:
        u, v = repeated_seq_nums_i.split('-')[:-1]
        shutil.copyfile(os.path.join(predict_path, '{}-{}-SDP.txt'.format(u,v)),os.path.join(predict_path,f'{repeated_seq_nums_i}.txt'))

    sub_dir = 'MOT17/train'
    seq_nums = os.listdir('dataset/MOT17/train')
    accs = []
    seqs = []
    for seq_num in seq_nums:
        shutil.copyfile(os.path.join(mot_path, sub_dir, f'{seq_num}/gt/gt.txt'),os.path.join(predict_path,f'{seq_num}.txt'))