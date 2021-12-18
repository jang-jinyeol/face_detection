import cv2
import os
import sys
sys.path.append('.')
import logging.config
logging.config.fileConfig("config/logging.conf")
logger = logging.getLogger('api')
from core.model_loader.face_detection.FaceDetModelLoader import FaceDetModelLoader
from core.model_handler.face_detection.FaceDetModelHandler import FaceDetModelHandler

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))

from backbone.backbone_def import BackboneFactory
from head.head_def import HeadFactory
from training_mode.conventional_training.train import FaceModel
from PIL import Image
import yaml

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
    model_name =  model_conf[scene][model_category]

    logger.info('Start to load the face detection model...')
    # load model
    try:
        faceDetModelLoader = FaceDetModelLoader(model_path, model_category, model_name)
    except Exception as e:
        logger.error('Failed to parse model configuration file!')
        logger.error(e)
        sys.exit(-1)
    else:
        logger.info('Successfully parsed the model configuration file model_meta.json!')

    try:
        model, cfg = faceDetModelLoader.load_model()
    except Exception as e:
        logger.error('Model loading failed!')
        logger.error(e)
        sys.exit(-1)
    else:
        logger.info('Successfully loaded the face detection model!')

    faceDetModelHandler = FaceDetModelHandler(model, 'cuda:0', cfg)
    image = cv2.imread("C:/Users/ddcfd/Downloads/test/arin_label.jpg", cv2.IMREAD_COLOR)

# model.eval()

# dets = faceDetModelHandler.inference_on_image(image)
# Fac