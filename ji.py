from __future__ import print_function

import logging as log
import os
import pathlib
import json
import cv2
import numpy as np
from openvino.inference_engine import IENetwork, IECore
import torch
import torchvision
import time



def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def non_max_suppression(prediction, conf_thres=0.1, iou_thres=0.6, merge=False, classes=None, agnostic=False):
    """Performs Non-Maximum Suppression (NMS) on inference results

    Returns:
         detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """
    prediction=torch.from_numpy(prediction)
    if prediction.dtype is torch.float16:
        prediction = prediction.float()  # to FP32

    nc = prediction[0].shape[1] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 300  # maximum number of detections per image
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)

    t = time.time()
    output = [None] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # If none remain process next image
        n = x.shape[0]  # number of boxes
        if not n:
            continue

        # Sort by confidence
        # x = x[x[:, 4].argsort(descending=True)]

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.boxes.nms(boxes, scores, iou_thres)
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            try:  # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
                iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
                weights = iou * scores[None]  # box weights
                x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
                if redundant:
                    i = i[iou.sum(1) > 1]  # require redundancy
            except:  # possible CUDA error https://github.com/ultralytics/yolov3/issues/1139
                print(x, i, x.shape, i.shape)
                pass

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            break  # time limit exceeded

    return output



device = 'CPU'
input_h, input_w, input_c, input_n = (480, 480, 3, 1)
log.basicConfig(level=log.DEBUG)

# For objection detection task, replace your target labels here.
label_id_map = {
    0: "fire",
}
exec_net = None


def init():
    """Initialize model

    Returns: model

    """
    #model_xml = "/project/train/src_repo/yolov5/runs/exp0/weights/best.xml"
    model_xml = "/usr/local/ev_sdk/model/openvino/yolov5x_10_best.xml"
    if not os.path.isfile(model_xml):
        log.error(f'{model_xml} does not exist')
        return None
    model_bin = pathlib.Path(model_xml).with_suffix('.bin').as_posix()
#     log.info("Loading network files:\n\t{}\n\t{}".format(model_xml, model_bin))
    net = IENetwork(model=model_xml, weights=model_bin)

    # Load Inference Engine
#     log.info('Loading Inference Engine')
    ie = IECore()
    global exec_net
    exec_net = ie.load_network(network=net, device_name=device)
#     log.info('Device info:')
#     versions = ie.get_versions(device)
#     print("{}".format(device))
#     print("MKLDNNPlugin version ......... {}.{}".format(versions[device].major, versions[device].minor))
#     print("Build ........... {}".format(versions[device].build_number))

    input_blob = next(iter(net.inputs))
    n, c, h, w = net.inputs[input_blob].shape
    global input_h, input_w, input_c, input_n
    input_h, input_w, input_c, input_n = h, w, c, n

    return net


def process_image(net, input_image):
    """Do inference to analysis input_image and get output

    Attributes:
        net: model handle
        input_image (numpy.ndarray): image to be process, format: (h, w, c), BGR
        thresh: thresh value

    Returns: process result

    """
    if not net or input_image is None:
        log.error('Invalid input args')
        return None
#     log.info(f'process_image, ({input_image.shape}')
    ih, iw, _ = input_image.shape

    # --------------------------- Prepare input blobs -----------------------------------------------------
    if ih != input_h or iw != input_w:
        input_image = cv2.resize(input_image, (input_w, input_h))
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    input_image = input_image/255
    input_image = input_image.transpose((2, 0, 1))
    images = np.ndarray(shape=(input_n, input_c, input_h, input_w))
    images[0] = input_image

    input_blob = next(iter(net.inputs))
    out_blob = next(iter(net.outputs))

    # --------------------------- Prepare output blobs ----------------------------------------------------
#     log.info('Preparing output blobs')
#     log.info(f"The output_name{net.outputs}")
    #print(net.outputs)
#     output_name = "Transpose_305"
#     try:
#         output_info = net.outputs[output_name]
#     except KeyError:
#         log.error(f"Can't find a {output_name} layer in the topology")
#         return None

#     output_dims = output_info.shape
#     log.info(f"The output_dims{output_dims}")
#     if len(output_dims) != 4:
#         log.error("Incorrect output dimensions for yolo model")
#     max_proposal_count, object_size = output_dims[2], output_dims[3]

#     if object_size != 7:
#         log.error("Output item should have 7 as a last dimension")

    #output_info.precision = "FP32"

    # --------------------------- Performing inference ----------------------------------------------------
#     log.info("Creating infer request and starting inference")
    res = exec_net.infer(inputs={input_blob: images})

    # --------------------------- Read and postprocess output ---------------------------------------------
#     log.info("Processing output blobs")

#     res = res[out_blob]
    data = res[out_blob]

    
    data=non_max_suppression(data, 0.4, 0.5)
    detect_objs = []
    if data[0]==None:
        return json.dumps({"objects": detect_objs})
    else:
        data=data[0].numpy()
        for proposal in data:
            if proposal[4] > 0 :
                confidence = proposal[4]
                xmin = np.int(iw * (proposal[0]/480))
                ymin = np.int(ih * (proposal[1]/480))
                xmax = np.int(iw * (proposal[2]/480))
                ymax = np.int(ih * (proposal[3]/480))
    #             if label not in label_id_map:
    #                 log.warning(f'{label} does not in {label_id_map}')
    #                 continue
                detect_objs.append({
                    'name': label_id_map[0],
                    'xmin': int(xmin),
                    'ymin': int(ymin),
                    'xmax': int(xmax),
                    'ymax': int(ymax),
                    'confidence': float(confidence)
                })
        return json.dumps({"objects": detect_objs})


if __name__ == '__main__':
    # Test API
    img = cv2.imread('/home/data/19/2209a117.jpg')
    predictor = init()
    result = process_image(predictor, img)
    log.info(result)
    



    
