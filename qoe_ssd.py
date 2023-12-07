import os, csv, json
import cv2
import tqdm
import numpy as np
import argparse
from collections import  defaultdict
import multiprocessing as mp
from more_itertools import chunked

parser = argparse.ArgumentParser()
parser.add_argument("--gtdir", type=str, default="runs/detect/exp/labels", help="ground truth dir")
parser.add_argument("--model", type=str)
args = parser.parse_args()


def detect():
    import tensorflow as tf
    import tensorflow_hub as hub
    if args.model:
        model = hub.KerasLayer(args.model)
    else:
        model = hub.KerasLayer("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")
    

    for file in tqdm.tqdm(os.listdir("frames")):
        fn_no_ext = file.split(".")[0]
        annotations = []
        with open(f"{args.gtdir}/{fn_no_ext}.txt") as f:
            for line in f:
                if line.strip().split()[0] != "0":
                    continue
                center_x, center_y, width, height = list(map(float, line.strip().split()[1:]))
                img = cv2.imread(os.path.join("frames", file))  # HWC
                center_x *= img.shape[1]
                center_y *= img.shape[0]
                width *= img.shape[1]
                height *= img.shape[0]
                annotations.append([f"frames/{file}", int(center_x), int(center_y), int(width), int(height)])

        
        all_detections = []
        for fn, center_x, center_y, width, height in annotations:
            gt_xmin = int(center_x) - int(width) / 2
            gt_ymin = int(center_y) - int(height) / 2
            gt_xmax = int(center_x) + int(width) / 2
            gt_ymax = int(center_y) + int(height) / 2
            img = cv2.imread(fn)
            H, W, C = img.shape
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.reshape(cv2.resize(img, (224, 224)), [1, 224, 224, 3])
            y = model(img)
            # ['detection_boxes', 'detection_anchor_indices', 'detection_multiclass_scores', 'num_detections', 'raw_detection_scores', 'raw_detection_boxes', 'detection_classes', 'detection_scores']
            selected_box = y["detection_boxes"][tf.logical_and(y["detection_scores"] > 0.5, y["detection_classes"] == 1)]
            selected_box = (selected_box.numpy() * np.array([H, W, H, W])).astype(np.int32)
            if not selected_box.size:
                continue
            ixmin = np.maximum(selected_box[:, 1], gt_xmin)
            iymin = np.maximum(selected_box[:, 0], gt_ymin)
            ixmax = np.minimum(selected_box[:, 3], gt_xmax)
            iymax = np.minimum(selected_box[:, 2], gt_ymax)
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih
            # person covers face, so use the gt area as union
            # uni = (gt_xmax - gt_xmin + 1.) * (gt_ymax - gt_ymin + 1.) + (selected_box[:, 2] - selected_box[:, 0] + 1.) * (selected_box[:, 3] - selected_box[:, 1] + 1.) - inters
            uni = (gt_xmax - gt_xmin + 1.) * (gt_ymax - gt_ymin + 1.)
            overlaps = inters / uni
            ovmax = np.max(overlaps)
            ymin, xmin, ymax, xmax = selected_box[np.argmax(overlaps)]
            if ovmax > 0.98:
                all_detections.append([fn, xmin, ymin, xmax, ymax])
            else:
                print(fn, ovmax)
            
        with open(f"output/{fn_no_ext}.detected.txt", 'w') as f:
            csv.writer(f).writerows(all_detections)



def voc_ap(rec, prec, use_07_metric=False):
    """Compute VOC AP given precision and recall. If use_07_metric is true, uses
    the VOC 07 11-point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap += p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def _analyze(subdir, overlap_thres=0.9):
    detections, annotations = {}, {}
    fns = []
    results = {}
    for file in os.listdir(subdir):
        with open(os.path.join(subdir, file)) as f:
            for fn, xmin, ymin, xmax, ymax in csv.reader(f):
                detections[fn] = [xmin, ymin, xmax, ymax]
    if not detections: 
        return
    for file in os.listdir("frames"):
        fn_no_ext = file.split(".")[0]
        with open(f"runs/detect/exp2/labels/{fn_no_ext}.txt") as f:
            for line in f:
                if line.strip().split()[0] != "0":
                    continue
                center_x, center_y, width, height = list(map(float, line.strip().split()[1:]))
                img = cv2.imread(os.path.join("frames", file))  # HWC
                center_x *= img.shape[1]
                center_y *= img.shape[0]
                width *= img.shape[1]
                height *= img.shape[0]
                gt_xmin = int(center_x) - int(width) / 2
                gt_ymin = int(center_y) - int(height) / 2
                gt_xmax = int(center_x) + int(width) / 2
                gt_ymax = int(center_y) + int(height) / 2
                annotations[f"frames/{file}"] = [gt_xmin, gt_ymin, gt_xmax, gt_ymax]
        fns.append(f"frames/{file}")
    fns.sort()
    print(len(fns))
    with open("annotations.json", "w") as f:
        json.dump(annotations, f, indent=2)
    with open("detections.json", "w") as f:
        json.dump(detections, f, indent=2)

    for inference_rate in np.arange(0.1, 1.1, 0.1):
        tp = np.zeros(len(fns))
        fp = np.zeros(len(fns))
        inferred_cnt = 0
        pre_res = None
        for idx, fn in enumerate(fns):
            gt_xmin, gt_ymin, gt_xmax, gt_ymax = annotations[fn]
            xmin, ymin, xmax, ymax = None, None, None, None
            # intersection = max(ixmax - ixmin + 1.0, 0.0) * max(iymax - iymin + 1., 0.)
            # union = (gt_xmax - gt_xmin + 1.) * (gt_ymax - gt_ymin + 1.) + (xmax - xmin + 1.0) * (ymax - ymin + 1.0) - intersection
            if fn not in annotations:
                fp[idx] = 1
                continue
            if fn in detections and 0 <= (idx+1) * inference_rate - inferred_cnt < 1:  # inference
                xmin, ymin, xmax, ymax = gt_xmin, gt_ymin, gt_xmax, gt_ymax
                inferred_cnt += 1
            elif pre_res:
                xmin, ymin, xmax, ymax = pre_res
            else:
                fp[idx] = 1
                continue
            pre_res = [xmin, ymin, xmax, ymax]
            ixmin = max(xmin, gt_xmin)
            iymin = max(ymin, gt_ymin)
            ixmax = max(xmax, gt_xmax)
            iymax = max(ymax, gt_ymax)
            intersection = max(ixmax - ixmin + 1.0, 0.0) * max(iymax - iymin + 1., 0.)
            union = (gt_xmax - gt_xmin + 1.) * (gt_ymax - gt_ymin + 1.) + (xmax - xmin + 1.0) * (ymax - ymin + 1.0) - intersection
            if intersection / union >= overlap_thres:
                tp[idx] = 1
            else:
                fp[idx] = 1
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(len(annotations))
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        results[inference_rate] = voc_ap(rec, prec)
    with open(f"analyzed.txt", "w") as f:
        json.dump(results, f, indent=2)        
        


def analyze_detect(overlap_thres = 0.9):
    detected_dir = [file.replace(".detected.txt", "") for file in os.listdir("results/inference/") if file.endswith("detected.txt")]
    if args.index == 0:
        detected_dirs = detected_dir[:len(detected_dir) // 2]
    elif args.index == 1:
        detected_dirs = detected_dir[len(detected_dir) // 2:]
    else:
        assert False, "invalid index"
    with mp.Pool(mp.cpu_count()) as pool:
        pool.map(_analyze, chunked(detected_dirs, 10))


def analyze_all():
    aps, weights = defaultdict(list), defaultdict(list)
    for file in os.listdir("results/analysis"):
        with open(f"results/analysis/{file}") as f:
            d = json.load(f)
        with open(f"frames/frame_images_DB/{file.replace('analyzed', 'labeled_faces')}") as f:
            weight = len(f.readlines())
            for k in d:
                aps[k].append(d[k])
                weights[k].append(weight)
    for k in aps:
        print(k, np.average(aps[k], weights=weights[k]))

detect()
_analyze("output")