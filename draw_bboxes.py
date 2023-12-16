import os
import cv2
import json
import pandas as pd

from datetime import datetime
from collections import Counter

ETALON_CSV_PATH = 'datasets/innopolis-high-voltage-challenge/sample_submission_1.csv'
TEST_IMAGES_PATH = 'datasets/innopolis-high-voltage-challenge'
PREDICTIONS_PATH = 'training_results/train/predictions.json'
CONF_THR = 0.01


def draw_from_json():
    f = open(PREDICTIONS_PATH)
    data = json.load(f)

    img_to_bboxes = {img: [] for img in images_list}
    for elem in data:
        if '_jpg' in elem['image_id']:
            img_name = elem['image_id'].split('_jpg')[0].replace('_JPG', '.JPG')
        elif '.rf' in elem['image_id']:
            img_name = elem['image_id'].split('.')[0].replace('_JPG', '.JPG')
        else:
            raise Exception
        # try:
            # elem['image_id'].rsplit('_')#[0][:-4]  # + '.JPG'
        # except KeyError:
        #     img_name = elem['image_id']  # + '.JPG'
        img_to_bboxes[img_name].append({'bbox': elem['bbox'], 'score': elem['score']})

    img_to_bboxes = dict(sorted(img_to_bboxes.items()))
    etalon_df_filenames = etalon_df.file_name.to_list()
    etalon_img_to_num_boxes = Counter([img_name[:-2] for img_name in etalon_df_filenames]).most_common()
    etalon_img_to_num_boxes = dict(sorted(dict(etalon_img_to_num_boxes).items()))
    df_file_name, df_x, df_y, df_w, df_h, df_probability = [], [], [], [], [], []
    for etalon_filename, num_boxes in etalon_img_to_num_boxes.items():
        im_name = etalon_filename + '.JPG'
        im_boxes = img_to_bboxes[im_name]
        im_full_name = os.path.join(TEST_IMAGES_PATH, im_name)
        im = cv2.imread(im_full_name)
        im_h, im_w, _ = im.shape

        box_num = 1
        while box_num < num_boxes + 1:
            for bbox in im_boxes:
                score = bbox['score']
                if score >= CONF_THR:
                    x1, y1, w, h = tuple(bbox['bbox'])
                    x2, y2 = x1 + w, y1 + h

                    df_file_name.append(im_name[:-4] + f'_{box_num}')
                    df_x.append(x1 / im_w)
                    df_y.append(y1 / im_h)
                    df_w.append(w / im_w)
                    df_h.append(h / im_h)
                    df_probability.append(score)

                    x1, y1, x2, y2 = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
                    cv2.rectangle(im, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)

                    bbox['score'] = 0
                    box_num += 1
                if box_num > num_boxes:
                    break

        cv2.imwrite(os.path.join(write_dir, f"{im_name[:-4]}.png"), im)

    df = pd.DataFrame()
    df['file_name'] = df_file_name
    df['x'] = df_x
    df['y'] = df_y
    df['w'] = df_w
    df['h'] = df_h
    df['probability'] = df_probability
    df.to_csv(os.path.join(write_dir, f'tweet_tweet_submission_{now}.csv'), index=False)


def draw_from_csv():
    bboxes_path = 'datasets/innopolis-high-voltage-challenge/sample_submission_1.csv'
    bboxes = pd.read_csv(bboxes_path)
    bboxes_file_name = bboxes.file_name.to_list()

    i = 0
    for img_path in images_list:
        img = cv2.imread(os.path.join(TEST_IMAGES_PATH, img_path))
        img_name = img_path[:-4]
        for bb_filename in bboxes_file_name:
            if img_name in bb_filename:
                img_h, img_w, _ = img.shape
                row = bboxes[bboxes['file_name'] == bb_filename]
                x1, y1 = row.x.to_list()[0], row.y.to_list()[0]
                x2, y2 = x1 + row.w.to_list()[0], y1 + row.h.to_list()[0]
                x1, y1, x2, y2 = int(round(img_w * x1)), int(round(img_h * y1)), int(round(img_w * x2)), int(
                    round(img_h * y2))
                cv2.rectangle(img, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)
                cv2.imwrite(f"{img_name}_with_bb_{i}.png", img)
                # cv2.imshow("lalala", img)
                # k = cv2.waitKey(0)
                i += 1


if __name__ == '__main__':
    images_list = os.listdir(TEST_IMAGES_PATH)
    etalon_df = pd.read_csv(ETALON_CSV_PATH)

    now = datetime.strftime(datetime.now(), '%d%m%y_%H%M%S')
    write_dir = f'submission_results/{now}'
    if not os.path.exists(write_dir):
        os.makedirs(write_dir)

    draw_from_json()
