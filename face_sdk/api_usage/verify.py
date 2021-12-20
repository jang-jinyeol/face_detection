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
    # face_cropper = FaceRecImageCropper()

    # t = transforms.Compose([transforms.ToTensor(),
    #                         transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])





    # image = cv2.imread("D:/test/arin_label.jpg", cv2.IMREAD_COLOR)
    # # capp = cv2.VideoCapture("C:/Users/ddcfd/Downloads/test/arin_label.jpg", cv2.IMREAD_COLOR)
    # cap2 = cv2.VideoCapture("http://192.168.0.75:4747/mjpegfeed?640x480")
    cap2 = cv2.VideoCapture(0)
    #
    #
    # # image2=image.copy()
    # bboxs = faceDetModelHandler.inference_on_image(image)
    # for box in bboxs:
    #
    #     det = np.asarray(list(map(int, box[0:4])), dtype=np.int32)
    # landmarks = faceAlignModelHandler.inference_on_image(image, det)
    # # for (x, y) in landmarks.astype(np.int32):
    # #     cv2.circle(image2, (x, y), 2, (255, 0, 0), -1)
    # kkk=faceAlignModelHandler.just_resize(image,det)
    # # t = transforms.Compose([transforms.ToTensor(),
    # #                         transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])
    # t = transforms.Compose([transforms.ToTensor()])
    # # # img_tensor = t(kkk).unsqueeze(0).to(device='cuda')
    # modela = Resnet(50, 0.4, 'ir').cuda()
    # # modela.cuda()
    # modela.eval()
    # embeddings=[]
    # name=[]
    # embs=[]
    # names = ['Unknown']
    # print(kkk.shape)
    # embs.append(modela(t(kkk).to(device='cuda').unsqueeze(0)))
    # # cv2.imshow("ss",kkk)
    # # cv2.waitKey()
    # # cv2.destroyAllWindows()
    # embedding = torch.cat(embs).mean(0, keepdim=True)
    # # print("embedding: ", embedding)
    # # print("embedding type: ", type(embedding))
    # # print("embedding shape: ",embedding.shape)
    # embeddings.append(embedding)
    # embeddings = torch.cat(embeddings)
    #
    # names.append("test")
    # torch.save(embeddings,"D:/test/face.pth")
    # names = np.array(names)
    # np.save("D:/test/names",names)


    # update 및 순회
    modela = Resnet(50, 0.4, 'ir_se').cuda()
    # modela.load_state_dict(torch.load("C:/Users/jinyeol/Desktop/save2_backup/model_ir_se50.pth"))
    # modela.load_state_dict(torch.load("D:/out_dir/Epoch_0_batch_1249.pth"))

    modela.eval()
    embeddings =  []
    names = ['Unknown']
    t = transforms.Compose([transforms.ToTensor()])
    root_dir =Path("D:/test2")
    for path in root_dir.iterdir():
        if path.is_file():
            continue
        else:
            embs = []
            for file in path.iterdir():
                if not file.is_file():
                    continue
                else:
                    img=cv2.imread(str(file))
                    bboxs = faceDetModelHandler.inference_on_image(img)
                    for box in bboxs:
                        det = np.asarray(list(map(int, box[0:4])), dtype=np.int32)
                    r_size = faceAlignModelHandler.just_resize(img, det)
                    with torch.no_grad():
                        embs.append(modela(t(r_size).to(device='cuda').unsqueeze(0)))

        if len(embs) == 0:
            continue
        embedding = torch.cat(embs).mean(0,keepdim=True)
        embeddings.append(embedding)
        names.append(path.name)
    embeddings = torch.cat(embeddings)


    names = np.array(names)
    torch.save(embeddings,"D:/test2/face.pth")
    np.save("D:/test2/names.npy", names)







    def infer(faces,target_embs):

        embs2 = []

        embs2.append(modela(t(faces).to(device='cuda').unsqueeze(0)))
        source_embs = torch.cat(embs2)
        diff = source_embs.unsqueeze(-1) - target_embs.transpose(1,0).unsqueeze(0)
        dist = torch.sum(torch.pow(diff, 2), dim=1)
        minimum, min_idx = torch.min(dist, dim=1)
        min_idx[minimum > 10] = -1 # if no match, set idx to -1
        return min_idx, minimum

    def draw_box_name(bbox,name,frame):
        frame = cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 6)
        frame = cv2.putText(frame,
                            name,
                            (bbox[0], bbox[1]),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            2,
                            (0, 255, 0),
                            3,
                            cv2.LINE_AA)
        return frame



    # embeddings = torch.load('C:/Users/jinyeol/PycharmProjects/InsightFace_Pytorch/data/facebank/facebank.pth')
    # names = np.load('C:/Users/jinyeol/PycharmProjects/InsightFace_Pytorch/data/facebank/names.npy')

    while cap2.isOpened():
        isSuccess,frame = cap2.read()
        if isSuccess:
            try:
                bboxs = faceDetModelHandler.inference_on_image(frame)
                bboxs = bboxs.astype(int)
                # det = np.asarray(list(map(int, bboxs[0:4])), dtype=np.int32)
                # image_show = frame.copy()
                for idx,box in enumerate(bboxs):
                    det = np.asarray(list(map(int, box[0:4])), dtype=np.int32)
                    landmarks = faceAlignModelHandler.inference_on_image(frame, det)
                    r_frame = faceAlignModelHandler.just_resize(frame, det)
                    results, score = infer(r_frame,embeddings)
                    # frame = draw_box_name(box, names[results[idx] + 1], frame)
                    frame = draw_box_name(box, names[results[idx] + 1] + '_{:.2f}'.format(score[idx]), frame)

                    # for (x, y) in landmarks.astype(np.int32):
                    #     cv2.circle(image_show, (x, y), 2, (255, 0, 0), -1)

                # for box in bboxs:
                #     box = list(map(int, box))
                #     cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
            except:
                print("detect error")
            cv2.imshow("cap",frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break






# model.eval()

# dets = faceDetModelHandler.inference_on_image(image)
