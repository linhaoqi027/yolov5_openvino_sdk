# yolov5_openvino_sdk
an SDK about how to use openvino model transformed from yolov5

## Before Start
train your yolov5 model on your own dataset following [Train Custom Data](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data)

## export model to onnx，using export.py.
### before run export.py:
- **changeto torch==1.5.1,torchvision==0.6.1**
- **delete [yolo.py](https://github.com/linhaoqi027/yolov5_openvino_sdk/blob/master/yolov5/models/yolo.py), and rename [yolo_1.py](https://github.com/linhaoqi027/yolov5_openvino_sdk/blob/master/yolov5/models/yolo_1.py) to yolo.py**

then you can run export.py to export onnx model.

## install OPENVINO2020R4
you can install following https://bbs.cvmart.net/topics/3117


## Convert onnx to openvino
AS for how to use openvino to inference ,please refer to [SDK](https://github.com/linhaoqi027/yolov5_openvino_sdk/tree/master/SDK)
using (convert_openvino.sh)[https://github.com/linhaoqi027/yolov5_openvino_sdk/blob/master/SDK/convert_openvino.sh] to convert onnx to openvino.Then you can get openvino model .bin and .xml.

## use openvino model to inference picture.
This profile provide an SDK.
using (ji.py)[https://github.com/linhaoqi027/yolov5_openvino_sdk/blob/master/SDK/ji.py] to using transformed openvino model.
(WARNING:NMS is not included in openvino model.)
