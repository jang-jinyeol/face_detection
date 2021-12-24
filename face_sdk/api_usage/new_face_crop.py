import cv2
import os
import sys

sys.path.append('.')
import logging.config
logging.config.fileConfig("config/logging.conf")
logger = logging.getLogger('api')
from core.model_loader.face_detection.FaceDetModelLoader import FaceDetModelLoader
from core.model_handler.face_detection.FaceDetModelHandler import FaceDetModelHandler
from core.model_loader.face_alignment.FaceAlignModelLoader import FaceAlignModelLoader
from core.image_cropper.arcface_cropper.FaceRecImageCropper import FaceRecImageCropper

from core.model_handler.face_alignment.FaceAlignModelHandler import FaceAlignModelHandler
from torchvision.transforms.functional import to_pil_image
import torchvision
from torchvision import transforms
from pathlib import Path

import numpy as np

from PIL import Image
import yaml
import matplotlib.pyplot as plt
sys.path.append('..')
from backbone.backbone_def import BackboneFactory
from head.head_def import HeadFactory
from training_mode.conventional_training.train import FaceModel
from backbone.ResNets import Resnet
import torch

# sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

# backbone_factory = BackboneFactory("ResNet", "C:/Users/ddcfd/PycharmProjects/face_detection/training_mode/backbone_conf.yaml")

# head_factory = HeadFactory("ArcFace", "C:/Users/ddcfd/PycharmProjects/face_detection/training_mode/head_conf.yaml")

# model = FaceModel(backbone_factory, head_factory)




with open('config/model_conf.yaml') as f:
    model_conf = yaml.full_load(f)

if __name__ == '__main__':
    # common setting for all model, need not modify.
    model_path = 'models'

    # model setting, modified along with model
    scene = 'non-mask'
    model_category = 'face_detection'
    model_category2 = 'face_alignment'

    model_name =  model_conf[scene][model_category]
    model_name2 =  model_conf[scene][model_category2]


    logger.info('Start to load the face detection model...')
    # load model
    try:
        faceDetModelLoader = FaceDetModelLoader(model_path, model_category, model_name)
        faceAlignModelLoader = FaceAlignModelLoader(model_path, model_category2, model_name2)

    except Exception as e:
        logger.error('Failed to parse model configuration file!')
        logger.error(e)
        sys.exit(-1)
    else:
        logger.info('Successfully parsed the model configuration file model_meta.json!')

    try:
        model, cfg = faceDetModelLoader.load_model()
        model2, cfg2 = faceAlignModelLoader.load_model()

    except Exception as e:
        logger.error('Model loading failed!')
        logger.error(e)
        sys.exit(-1)
    else:
        logger.info('Successfully loaded the face detection model!')



    faceDetModelHandler = FaceDetModelHandler(model, 'cuda:0', cfg)
    faceAlignModelHandler = FaceAlignModelHandler(model2, 'cuda', cfg2)



    img_path = "C:/Users/jinyeol/Desktop/Training_backup"

    directory = "crop"


    for root,dirs,files in os.walk(img_path):

        pass_dir = root.split("\\")

        if pass_dir[-1] == directory:
            continue

        for dir in dirs:
            if dir == directory:
                continue
            if not os.path.exists(root +"/"+ dir + "/" + directory):
                os.makedirs(root+"/" + dir + "/" + directory)

        for file in files:
            if len(file)>0:


                image = cv2.imread(root + "/"+ file, cv2.IMREAD_COLOR)

                faceDetModelHandler = FaceDetModelHandler(model, 'cuda:0', cfg)


                try:
                    bboxs = faceDetModelHandler.inference_on_image(image)
                    bboxs = bboxs.astype(int)
                    for idx, box in enumerate(bboxs):
                        det = np.asarray(list(map(int, box[0:4])), dtype=np.int32)
                        landmarks = faceAlignModelHandler.inference_on_image(image, det)
                        r_frame = faceAlignModelHandler.just_resize(image, det)
                        save_path_img = root + "/" + directory + "/crop_" + file
                        cv2.imwrite(save_path_img,r_frame)


                except Exception as e:
                    logger.error('Face detection failed!')
                    logger.error(e)
                    sys.exit(-1)
                # else:
                #     logger.info('Successful face detection!')