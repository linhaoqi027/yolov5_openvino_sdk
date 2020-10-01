# #!/bin/bash

# # Root directory of the script
# SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# /opt/intel/openvino/deployment_tools/model_optimizer/mo_tf.py \
# --input_model ${SCRIPT_DIR}/ssd_inception_v2.pb \
# --transformations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json \
# --tensorflow_object_detection_api_pipeline_config ${SCRIPT_DIR}/ssd_inception_v2_coco.config \
# --output_dir ${SCRIPT_DIR}/openvino \
# --model_name ssd_inception_v2 \
# --input image_tensor


# Root directory of the script




#source /opt/intel/openvino/deployment_tools/model_optimizer/venv/bin/activate
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

cd /usr/local/ev_sdk/yolov5
python models/export.py --weights ${SCRIPT_DIR}/yolov5x_10_best.pt --img 480 --batch 1

python3 /opt/intel/openvino_2020.4.287/deployment_tools/model_optimizer/mo.py \
--input_model ${SCRIPT_DIR}/yolov5x_10_best.onnx \
--output_dir ${SCRIPT_DIR}/openvino \
--input_shape [1,3,480,480]

