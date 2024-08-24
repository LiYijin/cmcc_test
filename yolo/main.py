
import torch
from pathlib import Path
import json
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import onnxruntime as ort
from PIL import Image
import numpy as np
from utils.general import non_max_suppression
import numpy as np
import cv2
import os
import time

import argparse 

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--gpu_id', '-id', help='Specify gpu id', required=True)
args = parser.parse_args()

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True, stride=32):
    """Resizes and pads image to new_shape with stride-multiple constraints, returns resized image, ratio, padding."""
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)


def clip_coords(boxes, img_shape):
    """

    """
    boxes[:, 0].clip(0, img_shape[1])  # x1
    boxes[:, 1].clip(0, img_shape[0])  # y1
    boxes[:, 2].clip(0, img_shape[1])  # x2
    boxes[:, 3].clip(0, img_shape[0])  # x2


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    """
    Image.open return as (hight, wight) format
    but for cv2.read it return as (hight, wight) format
    """
    if ratio_pad is None:  # 从img0_shape中计算
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain=old/new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords

if not os.path.exists("/dataset/annotations_trainval2017.zip"):
    os.system(
        "wget -o /dataset/annotations_trainval2017.zip http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    )
else:
    print("Use local annotation file")

if not os.path.exists("/dataset/val2017.zip"):
    os.system(
        "wget -o /dataset/val2017.zip http://images.cocodataset.org/zips/val2017.zip"
    )
else:
    print("Use local val file")

if not os.path.exists("/dataset/annotations"):
    os.system(
        "unzip -d /dataset/ /dataset/annotations_trainval2017.zip"
    )
else:
    print("annotation file is already unziped")

if not os.path.exists("/dataset/val2017"):
    os.system(
        "unzip -d /dataset/ /dataset/val2017.zip"
    )
else:
    print("val file is already unziped")
import os
import multiprocessing
import time
def run(gpu_id):
    os.environ['MUSA_VISIBLE_DEVICES'] = str(gpu_id)
    sess = ort.InferenceSession("yolov5m-24-3-640-640-fp16.onnx", providers=[('MUSAExecutionProvider', {"prefer_nhwc": '1'})])
    print("The model expects input shape: ", sess.get_inputs()[0].shape)
    img_size_h = sess.get_inputs()[0].shape[2]
    img_size_w = sess.get_inputs()[0].shape[3]
    results = []
    coco = COCO("/dataset/annotations/instances_val2017.json")
    image_ids = coco.getImgIds()
    total_time = 0.0
    # warm up
    for i in range(100):
        random_input = np.random.randn(24, 3, 640, 640).astype(np.float16)
        input_name = sess.get_inputs()[0].name
        outputs = sess.run(None, {input_name: random_input})

    def infer(img_ids, size=24):
        total_time = 0.0
        size_bks = []
        imgs = np.array([])
        for img_id in img_ids:
            img_info = coco.loadImgs(img_id)[0]
            img_path = Path("/dataset/") / 'val2017' / img_info['file_name']
            img = cv2.imread(img_path)
            size_bk = img.shape
            img = letterbox(img, new_shape=640)[0]  # preprocess
            img = img[:, :, ::-1].transpose(2, 0, 1)
            img = np.ascontiguousarray(img).astype(np.float32)
            img /= 255.0
            img = np.expand_dims(img, axis=0)
            size_bks.append(size_bk)
            if imgs.size == 0:
                imgs = img
            else:
                # print(imgs.shape, img.shape)
                imgs = np.concatenate((imgs, img), axis=0)
                # print(imgs.shape)
        
        imgs = imgs.astype(np.float16)
        input_name = sess.get_inputs()[0].name
        start_time = time.time()
        outputs = sess.run(None, {input_name: imgs})
        end_time = time.time()
        total_time += (end_time - start_time)
        outputs = np.array(outputs).astype(np.float32)

        filterd_predictions = non_max_suppression(torch.tensor(outputs[0].astype(np.float32)), conf_thres = 0.001, iou_thres = 0.65, labels=[], multi_label=True, agnostic=False, max_det=300)
        for i in range(size):
            for det in filterd_predictions[i]:
                score = float(det[4])
                category_id = int(det[5])
                coco_id = coco.getCatIds()[category_id]
                
                # reshape to old size
                box = np.squeeze(scale_coords((640, 640), np.expand_dims(det[0:4], axis=0), size_bks[i]), axis=0)
                det[0:4] = torch.tensor(box)
            
                seg = det[:4].tolist()
                
                seg[2] -= seg[0]
                seg[3] -= seg[1]
                results.append({'image_id': img_ids[i],
                                'category_id': coco_id,
                                'bbox': seg,
                                'score': score})
        return total_time

    batch_num = 209
    image_add = image_ids[-1]
    for i in range(16):
        image_ids.append(image_add)


    # for i in tqdm(range(batch_num)):
    for i in range(batch_num):
        if i != 208:
            total_time += infer(img_ids=image_ids[24 * i: 24 * i + 24])
        else:
            total_time += infer(img_ids=image_ids[24 * i: 24 * i + 24], size=8)  

    with open('predictions-ort-{}.json'.format(gpu_id), 'w') as f:
        json.dump(results, f, indent=4)


    coco_results = coco.loadRes('predictions-ort-{}.json'.format(gpu_id))
    coco_eval = COCOeval(coco, coco_results, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    mAP_50 = coco_eval.stats[1]
    total = 5000
    batch_cnt = int(total / 24) + 1
    print('Device: {}\ndata type: fp16\ndataset size: {}\nrequired mAP: 62.00%, mAP: {:.2f}%\nbatch size is 24\nuse time: {:.2f} Seconds\nlatency: {:.2f}ms/batch\nthroughput: {:.2f} fps'.format(gpu_id, total, mAP_50 * 100, total_time, 1000.0 * total_time / batch_cnt, total / total_time))


def main():
    run(args.gpu_id)
if __name__ == "__main__":
    main()
