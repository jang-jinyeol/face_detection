"""
@author: JiXuan Xu, Jun Wang
@date: 20201023
@contact: jun21wangustc@gmail.com 
"""
import sys
sys.path.append('.')
import logging.config
logging.config.fileConfig("config/logging.conf")
logger = logging.getLogger('api')
import os
import yaml
import cv2
import numpy as np
from core.model_loader.face_alignment.FaceAlignModelLoader import FaceAlignModelLoader
from core.model_handler.face_alignment.FaceAlignModelHandler import FaceAlignModelHandler

with open('config/model_conf.yaml') as f:
    model_conf = yaml.full_load(f)

if __name__ == '__main__':
    # common setting for all model, need not modify.
    model_path = 'models'

    # model setting, modified along with model
    scene = 'non-mask'
    model_category = 'face_alignment'
    model_name =  model_conf[scene][model_category]

    logger.info('Start to load the face landmark model...')
    # load model
    try:
        faceAlignModelLoader = FaceAlignModelLoader(model_path, model_category, model_name)
    except Exception as e:
        logger.error('Failed to parse model configuration file!')
        logger.error(e)
        sys.exit(-1)
    else:
        logger.info('Successfully parsed the model configuration file model_meta.json!')

    try:
        model, cfg = faceAlignModelLoader.load_model()
    except Exception as e:
        logger.error('Model loading failed!')
        logger.error(e)
        sys.exit(-1)
    else:
        logger.info('Successfully loaded the face landmark model!')

    faceAlignModelHandler = FaceAlignModelHandler(model, 'cuda', cfg)

    # read image
    # image_path = 'api_usage/test_images/001.jpg'
    #
    # image_det_txt_path = 'api_usage/test_images/test1_detect_res.txt'
    # image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    # with open(image_det_txt_path, 'r') as f:
    #     lines = f.readlines()
    # try:
    #     for i, line in enumerate(lines):
    #         line = line.strip().split()
    #         det = np.asarray(list(map(int, line[0:4])), dtype=np.int32)
    #         landmarks = faceAlignModelHandler.inference_on_image(image, det)
    #
    #         save_path_img = 'api_usage/temp/test1_' + 'landmark_res' + str(i) + '.jpg'
    #         save_path_txt = 'api_usage/temp/test1_' + 'landmark_res' + str(i) + '.txt'
    #         image_show = image.copy()
    #         with open(save_path_txt, "w") as fd:
    #             for (x, y) in landmarks.astype(np.int32):
    #                 cv2.circle(image_show, (x, y), 2, (255, 0, 0),-1)
    #                 line = str(x) + ' ' + str(y) + ' '
    #                 fd.write(line)
    #         cv2.imwrite(save_path_img, image_show)
    # except Exception as e:
    #     logger.error('Face landmark failed!')
    #     logger.error(e)
    #     sys.exit(-1)
    # else:
    #     logger.info('Successful face landmark!')


    # 수정

    img_path = "C:/Users/ddcfd/Downloads/CASIA-WebFace2/CASIA-WebFace2"
    image_det_txt_path = 'api_usage/temp/test1_detect_res2.txt'
    bb_directory = "bbox"
    land_directory = "land"
    # with open(image_det_txt_path, 'r') as f:
    #     lines = f.readlines()
    buf = open(image_det_txt_path,'r')
    line =  buf.readline()

    for root, dirs, files in os.walk(img_path):
        pass_dir = root.split("\\")

        if pass_dir[-1] == bb_directory or pass_dir[-1] == land_directory:
            continue

        # for dir in dirs:
        #
        #     if dir == bb_directory or dir == land_directory:
        #         continue
        #     if not os.path.exists(root +"/"+ dir + "/" + land_directory):
        #         os.makedirs(root+"/" + dir + "/" + land_directory)


        for file in files:
            if len(file)>0:


                    try:
                        print(root+"/"+file)
                        line=line.split()

                        det = np.asarray(list(map(int, line[0:4])), dtype=np.int32)


                        image = cv2.imread(root + "/" + file, cv2.IMREAD_COLOR)
                        print(root+"/"+file)
                        landmarks = faceAlignModelHandler.inference_on_image(image, det)
                        save_path_img = root + "/" + land_directory + "/land_" + file
                        save_path_txt = 'api_usage/temp/test1_' + 'landmark_res' + '.txt'
                        # image_show = image.copy()
                        with open(save_path_txt, "a") as fd:
                            for (x, y) in landmarks.astype(np.int32):
                                # cv2.circle(image_show, (x, y), 2, (255, 0, 0), -1)
                                line = str(x) + ' ' + str(y) + ' '
                                fd.write(line)
                        open(save_path_txt, "a").write("\n")
                        # cv2.imwrite(save_path_img, image_show)
                        line = buf.readline()


                    except Exception as e:
                        logger.error('Face landmark failed!')
                        logger.error(e)
                        sys.exit(-1)
                    else:
                        logger.info('Successful face landmark!')