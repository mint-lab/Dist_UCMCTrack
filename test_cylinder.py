from test import cam_para_to_config, save_lookup_table, test_table

cam_para_to_config(cam_para_file="cam_para/MOT17/MOT17-04-SDP.txt", save_file="detector/config_mot17_04.json")
save_lookup_table('detector/config_mot17_04.json', image_step=10, save_prefix = "detector/data/")
test_table('detector/data/MOT17_04_screenshot.png', 'detector/config_mot17_04.json', camera_name='MOT17_04')

# cam_para_to_config(cam_para_file="cam_para/MOT15/PETS09-S2L1.txt", save_file="detector/config_mot15_pets09.json")
# save_lookup_table("detector/config_mot15_pets09.json", image_step=10, save_prefix = "detector/data/")
# test_table('detector/data/MOT15_PETS09S2L1.jpg', "detector/config_mot15_pets09.json", camera_name='PETS09-S2L1')