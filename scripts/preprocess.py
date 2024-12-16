import os
import cv2
import numpy as np
from tqdm import tqdm

def preprocess_images(input_dir, output_dir, img_size=224):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    for filename in tqdm(os.listdir(input_dir)):
        filepath = os.path.join(input_dir, filename)
        if not os.path.isfile(filepath):
            continue

        img = cv2.imread(filepath)
        if img is None:
            continue
        img = cv2.resize(img, (img_size, img_size))

        outputpath = os.path.join(output_dir, filename)
        cv2.imwrite(outputpath, img)

preprocess_images('data/real', 'data/preprocessed/real')
preprocess_images('data/fake', 'data/preprocessed/fake')