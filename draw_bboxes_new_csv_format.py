import os
import cv2
import json
import pandas as pd
import numpy as np

from datetime import datetime

ETALON_CSV_PATH = 'datasets/innopolis-high-voltage-challenge/download_submission.csv'
TEST_IMAGES_PATH = 'datasets/innopolis-high-voltage-challenge'
PREDICTIONS_PATH = 'training_results/train_aug/predictions.json'
# CONF_THR = 0.05  # 0.37787
# CONF_THR = 0.01  # 0.37642
# CONF_THR = 0.1  # 0.38025
# CONF_THR = 0.5  # 0.38025

CONF_THR = 0.005


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
        img_to_bboxes[img_name].append({'bbox': elem['bbox'], 'score': elem['score']})

    img_to_bboxes = dict(sorted(img_to_bboxes.items()))
    etalon_df_filenames = etalon_df.file_name.to_list()

    df_dict = {filename: [] for filename in etalon_df_filenames}

    for etalon_filename in etalon_df_filenames:
        im_name = etalon_filename + '.JPG'
        im_boxes = img_to_bboxes[im_name]
        im_full_name = os.path.join(TEST_IMAGES_PATH, im_name)
        im = cv2.imread(im_full_name)
        im_h, im_w, _ = im.shape

        for bbox in im_boxes:
            score = bbox['score']
            if score >= CONF_THR:
                x1, y1, w, h = tuple(bbox['bbox'])
                # x_c, y_c = x1 + w/2, y1 + h/2
                x_c, y_c = x1, y1
                df_dict[im_name[:-4]].append([x_c / im_w, y_c / im_h, w / im_w, h / im_h, score])

                x2, y2 = x1 + w, y1 + h
                x1, y1, x2, y2 = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
                cv2.rectangle(im, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)

        cv2.imwrite(os.path.join(write_dir, f"{im_name[:-4]}.png"), im)

    df = pd.DataFrame()
    df['file_name'] = list(df_dict.keys())
    df_x, df_y, df_w, df_h, df_probability = [], [], [], [], []
    for filename in df_dict:
        if len(df_dict[filename]) > 1:
            df_dict[filename] = np.asarray(df_dict[filename])
            x = ', '.join([str(el) for el in df_dict[filename][:, 0]])
            y = ', '.join([str(el) for el in df_dict[filename][:, 1]])
            w = ', '.join([str(el) for el in df_dict[filename][:, 2]])
            h = ', '.join([str(el) for el in df_dict[filename][:, 3]])
            score = ', '.join([str(el) for el in df_dict[filename][:, 4]])
        else:
            try:
                x, y, w, h, score = df_dict[filename][0]
            except IndexError:
                x, y, w, h, score = 0.0, 0.0, 0.0, 0.0, 0.0
        df_x.append(x)
        df_y.append(y)
        df_w.append(w)
        df_h.append(h)
        df_probability.append(score)
    df['x'] = df_x
    df['y'] = df_y
    df['w'] = df_w
    df['h'] = df_h
    df['probability'] = df_probability
    df.to_csv(os.path.join(write_dir, f'tweet_tweet_submission_{now}.csv'), index=False, float_format='%.16f')


if __name__ == '__main__':
    images_list = os.listdir(TEST_IMAGES_PATH)
    etalon_df = pd.read_csv(ETALON_CSV_PATH)

    now = datetime.strftime(datetime.now(), '%d%m%y_%H%M%S')
    write_dir = f'submission_results/{now}'
    if not os.path.exists(write_dir):
        os.makedirs(write_dir)

    draw_from_json()
