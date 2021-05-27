# coding=utf-8

# 加载一些基础包以及设置logger
import detectron2
from detectron2.utils.logger import setup_logger

setup_logger()

# 加载其它一些库
import numpy as np
import cv2
import os

# 加载相关工具
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2 import model_zoo

if __name__ == '__main__':
    # input_path = "input1.jpg"
    # input_path = "datasets/Edge_Dataset_Version4_no_rotated/test2017"
    input_path = "/media/user/A3CDDB6409D0FE9C/Data/IOTDataset_split/test2017"
    # input_path = "/media/user/A3CDDB6409D0FE9C/Data/IOTDataset_split/tt"
    # input_path = "/media/user/A3CDDB6409D0FE9C/Data/IOTDataset_split/train2017/afternoon_30.jpg"
    # input_path = "datasets/Edge_Dataset_Version1/test2017"
    output_path = "/media/user/A3CDDB6409D0FE9C/Data/IOT_result/result2_6000"
    count_num = 0
    for f in os.listdir(input_path):
        if f.endswith(".xml"):
            continue
        print("now scanning:",f)
        # 指定模型的配置配置文件路径及网络参数文件的路径
        # 对于像下面这样写法的网络参数文件路径，程序在运行的时候就自动寻找，如果没有则下载。
        # Instance segmentation model
        model_file_path = "COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"
        # model_weights = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"
        # model_weights = "datasets/Edge_Dataset_Version4_no_rotated/output_test/model_final.pth"
        model_weights = "/media/user/A3CDDB6409D0FE9C/Data/IOT_result/test1/model_final.pth"
        # 加载图片
        img = cv2.imread(input_path+"/"+f)

        cur_output_path = os.path.join(output_path,"result" + str(count_num) +".jpg")
        count_num += 1

        # 创建一个detectron2配置
        cfg = get_cfg()
        # 要创建的模型的名称
        cfg.merge_from_file(model_zoo.get_config_file(model_file_path))
        cfg.SOLVER.IMS_PER_BATCH = 4
        ITERS_IN_ONE_EPOCH = int(4000 / cfg.SOLVER.IMS_PER_BATCH)
        # MAX ITERATION
        cfg.SOLVER.MAX_ITER = (ITERS_IN_ONE_EPOCH * 1) - 1  # 12 epochs
        # 保存模型文件的命名数据减1
        cfg.SOLVER.CHECKPOINT_PERIOD = ITERS_IN_ONE_EPOCH - 1
        # 迭代到指定次数，进行一次评估
        cfg.TEST.EVAL_PERIOD = ITERS_IN_ONE_EPOCH
        cfg.DATASETS.TRAIN = ("coco_my_train",)
        cfg.DATASETS.TEST = ("coco_my_val",)
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
        # cfg.OUTPUT_DIR = "./output_for_hole1"
        # 为模型设置阈值
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8
        #TODO: take correct params for the nms thresh!
        cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.3
        # 加载模型需要的数据
        cfg.MODEL.WEIGHTS = model_weights

        # CLASSES_NAME = ["railing"]
        CLASSES_NAME = ["WiFi","BT","Others"]
        MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).set(thing_classes=CLASSES_NAME)

        # 基于配置创建一个默认推断
        predictor = DefaultPredictor(cfg)
        # 利用这个推断对加载的影像进行分析并得到结果
        # 对于输出结果格式可以参考这里https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
        outputs = predictor(img)

        # 控制台中输出一些结果
        print(outputs["instances"].pred_classes)
        print(outputs["instances"].pred_boxes)
        total_num = len(outputs["instances"].pred_classes)
        with open("/media/user/A3CDDB6409D0FE9C/Data/IOT_result/result_files/" + f[:-4] + ".txt",'a+') as f:
            for i in range(total_num):
                f.write(str(outputs["instances"].pred_classes[i].to("cpu").numpy().tolist()) + "\t")
                # for pred_box in outputs["instances"].pred_boxes:
                pred_boxes = outputs["instances"].pred_boxes
                pred_box = pred_boxes[i].tensor[0]
                print(pred_box)
                minx = pred_box[0].to("cpu").numpy().tolist()
                miny = pred_box[1].to("cpu").numpy().tolist()
                maxx = pred_box[2].to("cpu").numpy().tolist()
                maxy = pred_box[3].to("cpu").numpy().tolist()
                # f.write(str(minx/875))
                f.write(str(minx)+"\t")
                f.write(str(maxx)+"\t")
                f.write(str(miny)+"\t")
                f.write(str(maxy)+"\n")
                    # break
            # f.write(str(outputs["instances"].pred_classes)+"\n")
            # f.write(str(outputs["instances"].pred_boxes))

        if len(outputs["instances"].pred_classes) == 0:
            continue
        else:
            print("success:",f)
        # 得到结果后可以使用Visualizer对结果可视化
        # img[:, :, ::-1]表示将BGR波段顺序变成RGB
        # scale表示输出影像的缩放尺度，太小了会导致看不清
            v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
            v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            # 获得绘制的影像
            result = v.get_image()[:, :, ::-1]
            # 将影像保存到文件
            print(cur_output_path)
            cv2.imwrite(cur_output_path, result)
            # break