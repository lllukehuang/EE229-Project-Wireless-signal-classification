# Apply Yolo v4 on WSAD

The main part of the code references the repo https://github.com/AlexeyAB/darknet

To simply train a Yolo v4 model, first download our dataset from https://jbox.sjtu.edu.cn/l/oFjsbK. And put the data and label files under the directory

```bash
./yolov4/build/darknet/x64/data/obj
```
Then run the python script to the train test split

```bash
$> python get_train_test_split.py
```

Then go back to the root dir and compile the code using
```bash
$> make
```
To train a new model, just run the script
```bash
$> bash yolo-obj.sh
```
The trained model will be saved in the /backup dir. To get the mAP result, just run
```bash
$> bash yolo-obj-map.sh
```
To get the output bounding box, just run
```bash
$> bash yolo-obj-test.sh
```
The log will be saved as result.txt, and the output bounding box will be saved under ./output dir.