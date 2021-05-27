#!/bin/bash

OPT = ""
OPT+="./build/darknet/x64/data/obj.data "
OPT+="./cfg/yolo-obj.cfg "
OPT+="./backup/best.weights "
OPT+="-dont_show "
./darknet detector map $OPT

