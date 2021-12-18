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
from core.model_handler.face_alignment.FaceAlignModelHandler import FaceAlignModelHandler

import numpy as np
# sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))

# from backbone.backbone_def import BackboneFactory
# from head.head_def import HeadFactory
# from training_mode.conventional_training.train import FaceModel
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


    # image = cv2.imread("C:/Users/ddcfd/Downloads/test/arin_label.jpg", cv2.IMREAD_COLOR)
    # capp = cv2.VideoCapture("C:/Users/ddcfd/Downloads/test/arin_label.jpg", cv2.IMREAD_COLOR)
    cap2 = cv2.VideoCapture("http://192.168.0.75:4747/mjpegfeed?640x480")
    while cap2.isOpened():
        # cv2.imshow("cap",cap2)
        isSuccess,frame = cap2.read()
        if isSuccess:
            bboxs = faceDetModelHandler.inference_on_image(frame)
            # det = np.asarray(list(map(int, bboxs[0:4])), dtype=np.int32)
            image_show = frame.copy()
            for box in bboxs:
                det = np.asarray(list(map(int, box[0:4])), dtype=np.int32)
                landmarks = faceAlignModelHandler.inference_on_image(frame, det)
                for (x, y) in landmarks.astype(np.int32):
                    cv2.circle(image_show, (x, y), 2, (255, 0, 0), -1)

            # for box in bboxs:
            #     box = list(map(int, box))
            #     cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
            cv2.imshow("cap",image_show)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break



    # image2=image.copy()
    # bboxs = faceDetModelHandler.inference_on_image(image)
    # for box in bboxs:
    #
    #     det = np.asarray(list(map(int, box[0:4])), dtype=np.int32)
    # landmarks = faceAlignModelHandler.inference_on_image(image, det)
    # for (x, y) in landmarks.astype(np.int32):
    #     cv2.circle(image2, (x, y), 2, (255, 0, 0), -1)
    #
    #
    # cv2.imshow("ss",image2)
    # cv2.waitKey()
    # cv2.destroyAllWindows()



# model.eval()

# dets = faceDetModelHandler.inference_on_image(image)
# Fac