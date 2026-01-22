import os.path
import sys; sys.path.append("../")
import argparse
import yaml
import time

import cv2
import numpy as np

import glob
import shutil
from tqdm import tqdm


def remove_markers_with_background(img, bg):
    
    if img is None or bg is None:
        print("Error: No se pudieron cargar las imágenes.")
        return

    # convert to float
    img_float = img.astype(np.float32)
    bg_float = bg.astype(np.float32) + 1e-5 

    # ratio image
    ratio = img_float / bg_float

    # normalize ratio image per channel
    scale_factor_ch1 = np.max(img_float[:, :, 0])
    scale_factor_ch2 = np.max(img_float[:, :, 1])
    scale_factor_ch3 = np.max(img_float[:, :, 2])

    ratio_visual = np.zeros_like(ratio)
    ratio_visual[:, :, 0] = (ratio[:, :, 0] * scale_factor_ch1).clip(0, 255).astype(np.uint8)
    ratio_visual[:, :, 1] = (ratio[:, :, 1] * scale_factor_ch2).clip(0, 255).astype(np.uint8)
    ratio_visual[:, :, 2] = (ratio[:, :, 2] * scale_factor_ch3).clip(0, 255).astype(np.uint8)
    ratio_visual = ratio_visual.astype(np.uint8)

    # bgr to gray
    gray_ratio = cv2.cvtColor(ratio_visual, cv2.COLOR_BGR2GRAY)
    
    # threshold
    _, mask = cv2.threshold(gray_ratio, 80, 255, cv2.THRESH_BINARY_INV) # Ajusta 80 si es necesario

    
    # dilate
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask_dilated = cv2.dilate(mask, kernel, iterations=3)

    # inpaint
    ratio_clean = cv2.inpaint(ratio_visual, mask_dilated, 5, cv2.INPAINT_TELEA)

    # convert to float
    ratio_clean_float = np.zeros_like(ratio_clean, dtype=np.float32)

    # divide by scale factor
    ratio_clean_float[:, :, 0] = ratio_clean[:, :, 0].astype(np.float32) / scale_factor_ch1
    ratio_clean_float[:, :, 1] = ratio_clean[:, :, 1].astype(np.float32) / scale_factor_ch2
    ratio_clean_float[:, :, 2] = ratio_clean[:, :, 2].astype(np.float32) / scale_factor_ch3

    # multiply by background
    final_float = ratio_clean_float * bg_float
    
    # clip and convert to uint8
    final_img = final_float.clip(0, 255).astype(np.uint8)

    return final_img



def main():

    # create dataset
    input_path = "../../data/v24_labels_24_32/train/"
    dataset = glob.glob(os.path.join(input_path, "*.npy"))

    bg = cv2.imread("nocontact.png")

    output_path = "../../data/v24_labels_24_32/dataset/train/"
    descartes_path = "../../data/v24_labels_24_32/descartes/"
    
    cont_name = 0

    for data_path in tqdm(dataset):
        
        data = np.load(data_path, allow_pickle=True).item()
        
        # show image and make it bigger
        gs_img = data["gs_img"]
        gs_img = cv2.cvtColor(gs_img, cv2.COLOR_BGR2RGB)
        gs_img = cv2.resize(gs_img, (320, 240))
        #gs_img = gs_img*255.0
        
        # remove markers with background
        gs_img = remove_markers_with_background(gs_img, bg)

        cv2.imshow("GelSight Mini", cv2.resize(gs_img, (int(320*1.5), int(240*1.5))))
        
        if cv2.waitKey(0) & 0xFF == ord("s"):
            shutil.move(data_path, output_path)
        else:
            shutil.move(data_path, descartes_path)


if __name__ == "__main__":

    main()
