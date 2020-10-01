# yolov5_openvino_sdk
an SDK about how to use openvino model transformed from yolov5

## Before Start
train your yolov5 model on your own dataset following [Train Custom Data](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data)

## export model to onnx，using export.py.
### before run export.py:
- **changeto torch==1.5.1,torchvision==0.6.1**
- **delete yolo.py, and rename yolo_1.py to yolo.py**

then you can run export.py to export onnx model.

## install OPENVINO2020R4
you can install following https://bbs.cvmart.net/topics/3117

## convert onnx to openvino through
```bash
python3 /opt/intel/openvino_2020.4.287/deployment_tools/model_optimizer/mo.py \
--input_model {input_dir}/yolov5s.onnx \
--output_dir {output_dir}  \
--input_shape [1,3,480,480]
```
 then you can get openvino model .bin and .xml.

## WARNING:NMS is not included in openvino model.
AS for how to use openvino to inference ,please refer to 
