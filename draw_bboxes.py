import os
import cv2
import json
import numpy as np
import pandas as pd

from datetime import datetime

TEST_IMAGES_PATH = 'datasets/innopolis-high-voltage-challenge'
PREDICTIONS_PATH = 'training_results/train/predictions.json'
CONF_THR = 0.05


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

    i = 0
    df_file_name, df_x, df_y, df_w, df_h, df_probability = [], [], [], [], [], []
    for k, v in img_to_bboxes.items():
        if v:
            img_full_name = os.path.join(TEST_IMAGES_PATH, k)
            img = cv2.imread(img_full_name)
            img_h, img_w, _ = img.shape
            box_num = 1
            for elem in v:
                score = elem['score']
                if score >= CONF_THR:
                    x1, y1, w, h = tuple(elem['bbox'])
                    x2, y2 = x1 + w, y1 + h

                    df_file_name.append(k[:-4] + f'_{box_num}')
                    df_x.append(x1 / img_w)
                    df_y.append(y1 / img_h)
                    df_w.append(w / img_w)
                    df_h.append(h / img_h)
                    df_probability.append(score)
                    box_num += 1

                    x1, y1, x2, y2 = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
                    cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)

            cv2.imwrite(os.path.join(write_dir, f"{k[:-4]}__{i}.png"), img)
            i += 1

    df = pd.DataFrame()

    sorted_ids = np.argsort(df_file_name)
    df['file_name'] = np.asarray(df_file_name)[sorted_ids]
    df['x'] = np.asarray(df_x)[sorted_ids]
    df['y'] = np.asarray(df_y)[sorted_ids]
    df['w'] = np.asarray(df_w)[sorted_ids]
    df['h'] = np.asarray(df_h)[sorted_ids]
    df['probability'] = np.asarray(df_probability)[sorted_ids]
    df.to_csv(os.path.join(write_dir, 'tweet_tweet_submission_v1.csv'), index=False)


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
    now = datetime.strftime(datetime.now(), '%d.%m.%y %H:%M:%S')
    write_dir = f'submission_results/{now}'
    if not os.path.exists(write_dir):
        os.makedirs(write_dir)

    draw_from_json()
