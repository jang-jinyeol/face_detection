"""
@author: Yinglu Liu, Jun Wang
@date: 20201012
@contact: jun21wangustc@gmail.com
"""

from face_masker import FaceMasker
import os

if __name__ == '__main__':
    # is_aug = False
    # image_path = 'Data/test-data/001.jpg'
    # face_lms_file = 'Data/test-data/test1_landmark_res0.txt'
    template_name = '3.png'
    # masked_face_path = 'test1_mask1.jpg'
    # face_lms_str = open(face_lms_file).readline().strip().split(' ')
    # face_lms = [float(num) for num in face_lms_str]
    # face_masker = FaceMasker(is_aug)
    # face_masker.add_mask_one(image_path, face_lms, template_name, masked_face_path)

    # 수정
    is_aug = False

    bb_directory = "bbox"
    land_directory = "land"
    masked = "masked"
    img_path = "C:/Users/jinyeol/Desktop/Arin"
    land_mark = "C:/Users/jinyeol/PycharmProjects/face_detection/face_sdk/api_usage/temp/test1_landmark_arin.txt"
    buf = open(land_mark, "r")
    line = buf.readline().strip().split()
    for root, dirs, files in os.walk(img_path):
        pass_dir = root.split("\\")
        if pass_dir[-1] == bb_directory or pass_dir[-1] == land_directory or pass_dir[-1] == masked:
            continue
        # for dir in dirs:
        #
        #     if dir == bb_directory or dir == land_directory or dir == masked:
        #         continue
        #     if not os.path.exists(root + "/" + dir + "/" + masked):
        #         os.makedirs(root + "/" + dir + "/" + masked)

        for file in files:
            if len(file) > 0:
                face_lms = [float(num) for num in line]
                face_masker = FaceMasker(is_aug)
                face_masker.add_mask_one(root+"/"+file, face_lms, template_name, root+"/"+"masked_"+file)

                line = buf.readline().strip().split()


