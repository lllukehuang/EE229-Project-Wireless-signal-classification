# 不知是否为detectron更新的原因，老版本训练代码无法运行
# 单独训练railing预测模型的代码
import logging
from collections import OrderedDict

import torch
from detectron2.evaluation import SemSegEvaluator, COCOEvaluator, COCOPanopticEvaluator, CityscapesInstanceEvaluator, \
    CityscapesSemSegEvaluator, PascalVOCDetectionEvaluator, LVISEvaluator, DatasetEvaluators
from detectron2.modeling import GeneralizedRCNNWithTTA
import detectron2.utils.comm as comm
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2
import random
import datetime
import time
import os
import sys

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultTrainer, launch, default_argument_parser, default_setup
from detectron2.config import get_cfg
from detectron2.utils.visualizer import ColorMode

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.evaluation import verify_results


class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can write your
    own training loop. You can use "tools/plain_train_net.py" as an example.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
                    ignore_label=cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                    output_dir=output_folder,
                )
            )
        if evaluator_type in ["coco", "coco_panoptic_seg"]:
            evaluator_list.append(COCOEvaluator(dataset_name, cfg, True, output_folder))
        if evaluator_type == "coco_panoptic_seg":
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        if evaluator_type == "cityscapes_instance":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesInstanceEvaluator(dataset_name)
        if evaluator_type == "cityscapes_sem_seg":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesSemSegEvaluator(dataset_name)
        elif evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        elif evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, cfg, True, output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res


def Train():
    # 把coco数据集转化成_DatasetCatalog样式并加metadata
    THING_CLASSES = ["WiFi", "BT", "Others"]
    register_coco_instances("custom", {},
                            # "../datasets/Edge_Dataset_Version4_no_rotated/annotations/instances_train2017.json", "../datasets/Edge_Dataset_Version4_no_rotated/train2017")
                            "/media/user/A3CDDB6409D0FE9C/Data/IOTDataset_split/annotations/instances_train2017.json", "/media/user/A3CDDB6409D0FE9C/Data/IOTDataset_split/train2017")
    register_coco_instances("custom1", {},
                            # "../datasets/Edge_Dataset_Version4_no_rotated/annotations/instances_val2017.json", "../datasets/Edge_Dataset_Version4_no_rotated/val2017")
                            "/media/user/A3CDDB6409D0FE9C/Data/IOTDataset_split/annotations/image_info_test2017.json", "/media/user/A3CDDB6409D0FE9C/Data/IOTDataset_split/test2017")
    # register_coco_instances("custom", {}, "/home/user/Data/coco2017/annotations_trainval2017/annotations/instances_train2017.json", "/home/user/Data/coco2017/train2017/train2017")
    custom_metadata = MetadataCatalog.get("custom")
    # dataset_dicts = DatasetCatalog.get("custom").set(thing_classes=THING_CLASSES)
    dataset_dicts = DatasetCatalog.get("custom")
    for d in random.sample(dataset_dicts, 3): # 随机抽样三个样本数据，检查一下样本有没有问题
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=custom_metadata, scale=0.5) # 创建一个看语义的对象
        vis = visualizer.draw_dataset_dict(d) # 把mask画上去
        cv2.imshow('Sample',vis.get_image()[:, :, ::-1]) # 展示图片
        cv2.waitKey()
    cv2.destroyAllWindows()


    cfg = get_cfg() # 获取config（默认参数值）
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"))
    # 从指定文件中merge config
    # 从一堆yaml格式结尾的文件中可以加载很多的congif
    # 这些config决定了网络的结构
    # 给定的config文件： _base_: Base-RCNN-FPN
        # 在base 的 config 中：
        # meta architecture: GeneralizedRCNN(定义在 meta_arch中)
            # 广义的RCNN模型，在这个里面主要定义的是神经网络一开始提取特征的部分（backbone）、建议框的生成、建议框里特征的提取和建议框的预测
        # backbone: build_resnet_fpn_backbone(定义在 backbone里的 fpn 和 resnet中)
            # 指定了特征提取后的金字塔处理和FPN输入管理(冻起来)
        # roi_headers: StandardROIHeads 任务与任务之间不共享特性
        # roi_box_header: FastRCNNConvFCHead conv+fc
        # roi_mask_head: MaskRCNNConvUpsampleHead conv + convtranspose + conv(1*1)
    # 设置、更改一些其他的参数
    cfg.DATASETS.TRAIN = ("custom",)
    # 指定训练集
    cfg.DATASETS.TEST = ("custom1",)
    cfg.DATALOADER.NUM_WORKERS = 4
    # 指定保存模型权重文件名称(设置断点续训!)
    # cfg.MODEL.WEIGHTS = 'model_final_maskrcnn.pkl'
    # cfg.MODEL.WEIGHTS = 'detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl'
    # images_per_batch
    cfg.SOLVER.IMS_PER_BATCH = 2
    # learning rate
    cfg.SOLVER.BASE_LR = 0.00025
    # 迭代次数
    cfg.SOLVER.MAX_ITER = (
        6000
    )
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (
        32
    )
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
    cfg.OUTPUT_DIR = "/media/user/A3CDDB6409D0FE9C/Data/IOT_result/test1"
    print(cfg)
    # 创建输出文件夹（这个在default设置里）
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    # trainer = DefaultTrainer(cfg)
    trainer = Trainer(cfg)
    # 把config加载到default trainer这个类里面取
    # default trainer是simple trainer的子类
    # 它的作用是
    # 1、从给定的config里面创建模型，优化器，调度程序scheduler（我猜这个是为了分布式计算搞得一个东西？）和dataloader
    # 2、加载最后一个检查点或者权重文件（cfg.MODEL.WEIGHTS）(如果有检查点和权重文件的话由resume_or_load函数搞定)
    # 3、根据config注册一些hooks
    model = trainer.build_model(cfg)
    trainer.test(cfg, model)
    trainer.resume_or_load(resume=False)
    print("begin to train")
    trainer.train()


# def Predict():
#     # register_coco_instances("custom", {}, "../datasets/Hole_Dataset_Version1/annotations/image_info_test2017.json", "datasets/Hole_Dataset_Version1/test2017")
#     register_coco_instances("custom", {}, "/media/user/A3CDDB6409D0FE9C/Data/IOTDataset_split/annotations/image_info_test2017.json", "/media/user/A3CDDB6409D0FE9C/Data/IOTDataset_split/test2017")
#     # register_coco_instances("custom", {}, "/home/user/Data/coco2017/annotations_trainval2017/annotations/instances_val2017.json",
#     #                         "/home/user/Data/coco2017/val2017")
#     custom_metadata = MetadataCatalog.get("custom")
#     DatasetCatalog.get("custom")
#
#
#     im = cv2.imread("../datasets/Hole_Dataset_Version1/test2017/10.jpg")
#     cfg = get_cfg()
#     cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"))
#     cfg.DATASETS.TEST = ("custom", )
#     cfg.OUTPUT_DIR = "./output_for_hole_1"
#     cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
#     cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
#     cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (
#         32
#     )
#     cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
#     predictor = DefaultPredictor(cfg)
#     outputs = predictor(im)
#     v = Visualizer(im[:, :, ::-1],
#                    metadata=custom_metadata,
#                    scale=1,
#                    instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
#     )
#     v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
#     cv2.imshow('Result',v.get_image()[:, :, ::-1])
#     cv2.waitKey()

def Predict(args):
    THING_CLASSES = ["WiFi", "BT", "Others"]
    register_coco_instances("custom", {},
                            # "../datasets/Edge_Dataset_Version4_no_rotated/annotations/instances_train2017.json", "../datasets/Edge_Dataset_Version4_no_rotated/train2017")
                            "/media/user/A3CDDB6409D0FE9C/Data/IOTDataset_split/annotations/instances_train2017.json",
                            "/media/user/A3CDDB6409D0FE9C/Data/IOTDataset_split/train2017")
    register_coco_instances("custom1", {},
                            # "../datasets/Edge_Dataset_Version4_no_rotated/annotations/instances_val2017.json", "../datasets/Edge_Dataset_Version4_no_rotated/val2017")
                            "/media/user/A3CDDB6409D0FE9C/Data/IOTDataset_split/annotations/image_info_test2017.json",
                            "/media/user/A3CDDB6409D0FE9C/Data/IOTDataset_split/test2017")
    # register_coco_instances("custom", {}, "/home/user/Data/coco2017/annotations_trainval2017/annotations/instances_train2017.json", "/home/user/Data/coco2017/train2017/train2017")
    custom_metadata = MetadataCatalog.get("custom")
    # dataset_dicts = DatasetCatalog.get("custom").set(thing_classes=THING_CLASSES)
    dataset_dicts = DatasetCatalog.get("custom")
    for d in random.sample(dataset_dicts, 3):  # 随机抽样三个样本数据，检查一下样本有没有问题
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=custom_metadata, scale=0.5)  # 创建一个看语义的对象
        vis = visualizer.draw_dataset_dict(d)  # 把mask画上去
        cv2.imshow('Sample', vis.get_image()[:, :, ::-1])  # 展示图片
        cv2.waitKey()
    cv2.destroyAllWindows()

    cfg = get_cfg()  # 获取config（默认参数值）
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"))
    # 从指定文件中merge config
    # 从一堆yaml格式结尾的文件中可以加载很多的congif
    # 这些config决定了网络的结构
    # 给定的config文件： _base_: Base-RCNN-FPN
    # 在base 的 config 中：
    # meta architecture: GeneralizedRCNN(定义在 meta_arch中)
    # 广义的RCNN模型，在这个里面主要定义的是神经网络一开始提取特征的部分（backbone）、建议框的生成、建议框里特征的提取和建议框的预测
    # backbone: build_resnet_fpn_backbone(定义在 backbone里的 fpn 和 resnet中)
    # 指定了特征提取后的金字塔处理和FPN输入管理(冻起来)
    # roi_headers: StandardROIHeads 任务与任务之间不共享特性
    # roi_box_header: FastRCNNConvFCHead conv+fc
    # roi_mask_head: MaskRCNNConvUpsampleHead conv + convtranspose + conv(1*1)
    # 设置、更改一些其他的参数
    cfg.DATASETS.TRAIN = ("custom",)
    # 指定训练集
    cfg.DATASETS.TEST = ("custom1",)
    cfg.DATALOADER.NUM_WORKERS = 4
    # 指定保存模型权重文件名称(设置断点续训!)
    # cfg.MODEL.WEIGHTS = 'model_final_maskrcnn.pkl'
    # cfg.MODEL.WEIGHTS = 'detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl'
    cfg.MODEL.WEIGHTS = "/media/user/A3CDDB6409D0FE9C/Data/IOT_result/test1/model_final.pth"
    # images_per_batch
    cfg.SOLVER.IMS_PER_BATCH = 2
    # learning rate
    cfg.SOLVER.BASE_LR = 0.00025
    # 迭代次数
    cfg.SOLVER.MAX_ITER = (
        6000
    )
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (
        32
    )
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
    cfg.OUTPUT_DIR = "/media/user/A3CDDB6409D0FE9C/Data/IOT_result/test1"
    print(cfg)
    # 创建输出文件夹（这个在default设置里）
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print(args)
    port = 2 ** 15 + 2 ** 14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    # launch(
    #     Train,
    #     num_gpus_per_machine = 2,
    #     num_machines = 1,
    #     machine_rank = 0,
    #     dist_url = "tcp://127.0.0.1:{}".format(port),
    # )
    # Train()
    Predict(args)