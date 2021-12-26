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
from backbone.MobileFaceNets import MobileFaceNet
import torch

# sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

backbone_factory = BackboneFactory("MobileFaceNet", "C:/Users/jinyeol/PycharmProjects/face_detection/training_mode/backbone_conf.yaml")

head_factory = HeadFactory("ArcFace", "C:/Users/jinyeol/PycharmProjects/face_detection/training_mode/head_conf.yaml")

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




    # image = cv2.imread("D:/test/arin_label.jpg", cv2.IMREAD_COLOR)
    # # capp = cv2.VideoCapture("C:/Users/ddcfd/Downloads/test/arin_label.jpg", cv2.IMREAD_COLOR)
    # cap2 = cv2.VideoCapture("http://192.168.0.75:4747/mjpegfeed?640x480")
    # cap2 = cv2.VideoCapture(0)
    #
    #
    # # image2=image.copy()
    # bboxs = faceDetModelHandler.inference_on_image(image)
    # for box in bboxs:

        # det = np.asarray(list(map(int, box[0:4])), dtype=np.int32)
    # landmarks = faceAlignModelHandler.inference_on_image(image, det)
    # for (x, y) in landmarks.astype(np.int32):
    #     cv2.circle(image2, (x, y), 2, (255, 0, 0), -1)
    # kkk=faceAlignModelHandler.just_resize(image,det)
    # t = transforms.Compose([transforms.ToTensor(),
    #                         transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])
    # t = transforms.Compose([transforms.ToTensor()])
    # # img_tensor = t(kkk).unsqueeze(0).to(device='cuda')
    # modela = Resnet(50, 0.4, 'ir').cuda()
    # modela.cuda()
    # modela.eval()
    # embeddings=[]
    # name=[]
    # embs=[]
    # names = ['Unknown']

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









    # 모델 및 파라미터 로드----------------------------------
    # modela = Resnet(50, 0.4, 'ir').cuda()
    modela = MobileFaceNet(512,7,7).cuda()


    # modela = FaceModel(backbone_factory).cuda()
    # modela.load_state_dict(torch.load("C:/Users/jinyeol/Desktop/ir50.pt"),strict=False)
    modela.load_state_dict(torch.load("C:/Users/jinyeol/Desktop/model_ir_se50.pth"),strict=False)
    modela.eval()

    # modelb = Resnet(50, 0.4, 'ir_se').cuda()
    # modelb.load_state_dict(torch.load("C:/Users/jinyeol/Desktop/model_ir_se50.pth"),strict=False)


    # 모델 정보------------------------------------------------------
    # modela = torch.load("C:/Users/jinyeol/Desktop/model_ir_se50.pth")
    # print(modela)

    # print("-------modela 정보 출력-------")
    # for param_tensor in onnx_model.state_dict():
    #     print(param_tensor, "/t", onnx_model.state_dict()[param_tensor].size())
    # print("-------------------------------")
    #
    # print("-------modelb 정보 출력-------")
    # for param_tensor in modelb.state_dict():
    #     print(param_tensor, "/t", modela.state_dict()[param_tensor].size())
    # print("-------------------------------")
    # modela_load = torch.load("C:/Users/jinyeol/Desktop/Epoch_9_batch_79.pth")
    # modelb_load = torch.load("C:/Users/jinyeol/Desktop/model_ir_se50.pth")

    # print("-------model a's epcoh & batch-------")
    # modela_epoch=modela_load['epoch']
    # modela_batch=modela_load['batch_id']
    # print(modela_epoch)
    # print(modela_batch)
    # print(modela_load)
    # print("-------------------------------")


    # print("----model b's epcoh & batch-------")
    # modelb_epoch=modelb_load['epoch']
    # modelb_batch=modelb_load['batch_id']
    # print(modelb_epoch)
    # print(modelb_batch)
    # print(modelb_load)
    # print("-------------------------------")





    # 임베딩-------------------------------------------
    #
    embeddings =  []
    names = ['Unknown']
    # Transform
    t = transforms.Compose([transforms.ToTensor(),
                            transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])
    root_dir =Path("C:/Users/jinyeol/Desktop/Training_backup(crop)")
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
    torch.save(embeddings,"C:/Users/jinyeol/Desktop/Training_backup(crop)/face.pth")
    np.save("C:/Users/jinyeol/Desktop/Training_backup(crop)/names.npy", names)







    def infer(faces,target_embs):

        embs2 = []
        modela.eval()

        embs2.append(modela(t(faces).to(device='cuda').unsqueeze(0)))
        source_embs = torch.cat(embs2)
        diff = source_embs.unsqueeze(-1) - target_embs.transpose(1,0).unsqueeze(0)
        dist = torch.sum(torch.pow(diff, 2), dim=1)
        minimum, min_idx = torch.min(dist, dim=1)
        print(minimum)
        min_idx[minimum > 1.5e-08] = -1 # if no match, set idx to -1
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


    # # ----------임베딩 로드 ----------------------------------------
    embeddings = torch.load('C:/Users/jinyeol/Desktop/Training_backup(crop)/face.pth')
    names = np.load('C:/Users/jinyeol/Desktop/Training_backup(crop)/names.npy')
    # embeddings = torch.load('C:/Users/jinyeol/Desktop/임베딩/학습시킨임베딩(CASIA)/face.pth')
    # names = np.load('C:/Users/jinyeol/Desktop/임베딩/학습시킨임베딩(CASIA)/names.npy')
    # #-------------------------------------------------------------------

    #------------------- 이미지 확인 --------------------------
    # img_path = "C:/Users/jinyeol/Desktop/Training_backup/Yua/357212_465303_379.jpg"

    # img_path = "C:/Users/jinyeol/Desktop/Training_backup/Yua/357212_465303_379.jpg"
    # img = cv2.imread(img_path)
    # bboxs = faceDetModelHandler.inference_on_image(img)
    # bboxs = bboxs.astype(int)
    # # det = np.asarray(list(map(int, bboxs[0:4])), dtype=np.int32)
    # # image_show = frame.copy()
    # for idx, box in enumerate(bboxs):
    #     det = np.asarray(list(map(int, box[0:4])), dtype=np.int32)
    #     landmarks = faceAlignModelHandler.inference_on_image(img, det)
    #     r_frame = faceAlignModelHandler.just_resize(img, det)
    #     results, score = infer(r_frame, embeddings)
    #     # frame = draw_box_name(box, names[results[idx] + 1], frame)
    #     img1 = draw_box_name(box, names[results[idx] + 1] + '_{:.2f}'.format(score[idx]), img)
    # cv2.imshow("Arin",img)
    # cv2.waitKey(0)
    #----------------------------------------------------------
    img_path2 = "C:/Users/jinyeol/Desktop/123.jpg"
    img = cv2.imread(img_path2)
    bboxs = faceDetModelHandler.inference_on_image(img)
    bboxs = bboxs.astype(int)
    # det = np.asarray(list(map(int, bboxs[0:4])), dtype=np.int32)
    # image_show = frame.copy()
    for idx, box in enumerate(bboxs):
        det = np.asarray(list(map(int, box[0:4])), dtype=np.int32)
        landmarks = faceAlignModelHandler.inference_on_image(img, det)
        r_frame = faceAlignModelHandler.just_resize(img, det)
        results, score = infer(r_frame, embeddings)
        # frame = draw_box_name(box, names[results[idx] + 1], frame)
        img2 = draw_box_name(box, names[results[idx] + 1] + '_{:.2f}'.format(score[idx]), img)

    # img1 = cv2.resize(img1, (600, 600))
    # img2 = cv2.resize(img2, (600, 600))

    # print(score)
    # addh = cv2.hconcat([img1, img2])
    # cv2.imshow("Result", addh)
    cv2.imshow("sds",img2)
    cv2.waitKey(0)



    #------------------- 캠 확인 --------------------------

    # cap2 = cv2.VideoCapture(0)
    #
    # while cap2.isOpened():
    #     isSuccess,frame = cap2.read()
    #     if isSuccess:
    #         try:
    #
    #             bboxs = faceDetModelHandler.inference_on_image(frame)
    #             bboxs = bboxs.astype(int)
    #             # det = np.asarray(list(map(int, bboxs[0:4])), dtype=np.int32)
    #             # image_show = frame.copy()
    #             for idx,box in enumerate(bboxs):
    #                 det = np.asarray(list(map(int, box[0:4])), dtype=np.int32)
    #                 landmarks = faceAlignModelHandler.inference_on_image(frame, det)
    #                 r_frame = faceAlignModelHandler.just_resize(frame, det)
    #                 results, score = infer(r_frame,embeddings)
    #                 # frame = draw_box_name(box, names[results[idx] + 1], frame)
    #                 frame = draw_box_name(box, names[results[idx] + 1] + '_{:.2f}'.format(score[idx]), frame)
    #
    #                 # for (x, y) in landmarks.astype(np.int32):
    #                 #     cv2.circle(image_show, (x, y), 2, (255, 0, 0), -1)
    #
    #             # for box in bboxs:
    #             #     box = list(map(int, box))
    #             #     cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
    #         except:
    #             print("detect error")
    #         cv2.imshow("cap",frame)
    #
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    #----------------------------------------------------------





# model.eval()

# dets = faceDetModelHandler.inference_on_image(image)
