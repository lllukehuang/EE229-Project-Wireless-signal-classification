#!/bin/bash

OPT = ""
OPT+="./build/darknet/x64/data/obj.data "
OPT+="./cfg/yolo-obj-test.cfg "
OPT+="./build/darknet/x64/yolov4.conv.137 "
#OPT+="./backup/yolo-obj_last.weights "
OPT+="-gpus 1,2 "
OPT+="-dont_show "
OPT+="-map "
./darknet detector train $OPT

