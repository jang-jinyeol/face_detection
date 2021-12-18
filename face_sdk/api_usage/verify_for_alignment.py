import cv2
import os
import sys
sys.path.append('.')
import logging.config
logging.config.fileConfig("config/logging.conf")
logger = logging.getLogger('api')
from core.model_loader.face_alignment.FaceAlignModelLoader import FaceAlignModelLoader
from core.model_handler.face_alignment.FaceAlignModelHandler import FaceAlignModelHandler
import yaml


with open('config/model_conf.yaml') as f:
    model_conf = yaml.full_load(f)

class face_alignment():
    def __init__(self):
        # common setting for all model, need not modify.
        model_path = 'models'

        # model setting, modified along with model
        scene = 'non-mask'
        model_category = 'face_alignment'
        model_name = model_conf[scene][model_category]
        print(model_name)
