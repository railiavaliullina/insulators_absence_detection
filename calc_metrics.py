import os
import cv2
import torch
import pandas as pd
from pprint import pprint
from torchmetrics.detection.mean_ap import MeanAveragePrecision


def get_gt():
    gt_boxes_path = '/Users/railiavaliullina/Documents/GitHub/insulators_absence_detection/datasets/4.v1i.yolov8/' \
                    'valid/labels'
    gt_list = [filename.split('_jpg')[0].replace('_JPG', '') if '_jpg' in filename
               else filename.split('.rf')[0].replace('_JPG', '') for filename in os.listdir(gt_boxes_path)]
    gts = {filename: [] for filename in gt_list}
    for label_path in os.listdir(gt_boxes_path):
        filename = label_path.split('_jpg')[0].replace('_JPG', '') if '_jpg' in label_path \
            else label_path.split('.rf')[0].replace('_JPG', '')
        with open(os.path.join(gt_boxes_path, label_path), 'r') as f:
            labels = f.readlines()
        for label in labels:
            x, y, w, h = tuple(label.split()[1:])
            x, y, w, h = float(x), float(y), float(w), float(h)
            gts[filename].append([x, y, w, h])
    return gts


def get_pred():
    pred_boxes_path = '/Users/railiavaliullina/Documents/GitHub/insulators_absence_detection/submission_results/' \
                      '171223_203406/tweet_tweet_submission_171223_203406.csv'
    preds_df = pd.read_csv(pred_boxes_path)
    preds = {filename[:-2]: [] for filename in preds_df.file_name.to_list()}
    for _, row in preds_df.iterrows():
        filename, x, y, w, h, probability = row
        preds[filename[:-2]].append([x, y, w, h])
    return preds


def scale_values(x, y, w, h, im_w, im_h):
    return x * im_w, y * im_h, w * im_w, h * im_h


def round_values(x_list):
    return tuple([int(round(el)) for el in x_list])


def draw_gt(filename, im, im_w, im_h):
    for box in gts[filename]:
        x_c, y_c, w, h = tuple(box)
        x_c, y_c, w, h = scale_values(x_c, y_c, w, h, im_w, im_h)
        x1, y1, x2, y2 = x_c - w / 2, y_c - h / 2, x_c + w / 2, y_c + h / 2
        x1, y1, x2, y2 = round_values([x1, y1, x2, y2])
        cv2.rectangle(im, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)
    return im


def draw_pred(filename, im, im_w, im_h):
    if filename in preds:
        for box in preds[filename]:
            x_c, y_c, w, h = tuple(box)
            x_c, y_c, w, h = scale_values(x_c, y_c, w, h, im_w, im_h)
            x1, y1, x2, y2 = x_c - w / 2, y_c - h / 2, x_c + w / 2, y_c + h / 2
            x1, y1, x2, y2 = round_values([x1, y1, x2, y2])
            cv2.rectangle(im, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
    return im


def draw_boxes():
    imgs_path = '/Users/railiavaliullina/Documents/GitHub/insulators_absence_detection/datasets/' \
                'innopolis-high-voltage-challenge'
    for filename in gts:
        if filename in preds:
            im = cv2.imread(os.path.join(imgs_path, filename) + '.JPG')
            im_h, im_w, _ = im.shape
            im = draw_gt(filename, im, im_w, im_h)
            im = draw_pred(filename, im, im_w, im_h)
            cv2.imwrite(f'{filename}_GT_PRED.png', im)


def get_map():
    for filename in gts:
        if filename in preds:
            gt_boxes = gts[filename]
            pred_boxes = preds[filename]

            gt_boxes_ = []
            for bb in gt_boxes:
                x1, y1, w, h = bb
                x2, y2 = x1 + w, y1 + h
                gt_boxes_.append([x1, y1, x2, y2])

            pred_boxes_ = []
            for bb in pred_boxes:
                x1, y1, w, h = bb
                x2, y2 = x1 + w, y1 + h
                pred_boxes_.append([x1, y1, x2, y2])

            pred = [dict(boxes=torch.tensor(pred_boxes_), scores=torch.tensor([1.0 for _ in range(len(pred_boxes_))]),
                         labels=torch.tensor([0 for _ in range(len(pred_boxes_))]), )]
            # target = [dict(boxes=torch.tensor(pred_boxes), scores=torch.tensor([1.0 for _ in range(len(pred_boxes))]),
            #              labels=torch.tensor([0 for _ in range(len(pred_boxes))]), )]
            target = [dict(boxes=torch.tensor(gt_boxes_), scores=torch.tensor([1.0 for _ in range(len(pred_boxes_))]),
                           labels=torch.tensor([0 for _ in range(len(gt_boxes_))]), )]

            metric = MeanAveragePrecision()
            metric.update(pred, target)
            pprint(metric.compute())


if __name__ == '__main__':
    gts = get_gt()
    preds = get_pred()

    draw_boxes()
    get_map()
