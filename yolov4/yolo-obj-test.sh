#!/bin/bash

OPT = ""
OPT+="./build/darknet/x64/data/obj.data "
OPT+="./cfg/yolo-obj.cfg "
OPT+="./backup/best.weights "
#OPT+="../data/* "
OPT+="-dont_show "
OPT+="-ext_output "

./darknet detector test $OPT < ./test1.txt >result1.txt

