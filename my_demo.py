import math
import os
import random

import cv2 as cv
import keras.backend as K
import numpy as np

# from data_generator import generate_trimap, random_choice, get_alpha_test
from model import build_encoder_decoder, build_refinement
from utils import compute_mse_loss, compute_sad_loss
from utils import get_final_output, safe_crop, draw_str
from config import unknown_code
import glob


kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
def get_alpha_test(name):
    alpha = cv.imread(name, 0)
    return alpha

def random_choice(trimap, crop_size=(320, 320)):
    crop_height, crop_width = crop_size
    y_indices, x_indices = np.where(trimap == unknown_code)
    num_unknowns = len(y_indices)
    x, y = 0, 0
    if num_unknowns > 0:
        ix = np.random.choice(range(num_unknowns))
        center_x = x_indices[ix]
        center_y = y_indices[ix]
        x = max(0, center_x - int(crop_width / 2))
        y = max(0, center_y - int(crop_height / 2))
    return x, y

def generate_trimap(alpha):
    fg = np.array(np.equal(alpha, 255).astype(np.float32))
    # fg = cv.erode(fg, kernel, iterations=np.random.randint(1, 3))
    unknown = np.array(np.not_equal(alpha, 0).astype(np.float32))
    unknown = cv.dilate(unknown, kernel, iterations=2)
    trimap = fg * 255 + (unknown - fg) * 128
    return trimap.astype(np.uint8)



def composite4(fg, bg, a, w, h):
    fg = np.array(fg, np.float32)
    bg_h, bg_w = bg.shape[:2]
    x = 0
    #if bg_w > w:
    #    x = np.random.randint(0, bg_w - w)
    y = 0
    #if bg_h > h:
   #     y = np.random.randint(0, bg_h - h)
    bg = np.array(bg[y:y + h, x:x + w], np.float32)
    alpha = np.zeros((h, w, 1), np.float32)
    alpha[:, :, 0] = a / 255.
    im = alpha * fg + (1 - alpha) * bg
    im = im.astype(np.uint8)
    return im, bg




if __name__ == '__main__':
    img_rows, img_cols = 320, 320
    channel = 4

    pretrained_path = 'models/final.42-0.0398.hdf5'
    encoder_decoder = build_encoder_decoder()
    final = build_refinement(encoder_decoder)
    final.load_weights(pretrained_path)
    print(final.summary())

    total_loss = 0.0
    
    alpha_file_list = glob.glob('trimap/*.jpg')
    bgr_file_list = glob.glob('ori/*.jpg')
    alpha_file_list = sorted(alpha_file_list)
    bgr_file_list = sorted(bgr_file_list)
    bg_file = '0002.png'
    
    for idx, (alpha_file, bgr_file) in enumerate(zip(alpha_file_list, bgr_file_list)):
        # print(alpha_file)
        # print(bgr_file)
        # res_img = matting(alpha_file, bgr_file, bg_file)
        print(idx)
     
        bgr_img = cv.imread(bgr_file)
        bg_h, bg_w = bgr_img.shape[:2]
        print('bg_h, bg_w: ' + str((bg_h, bg_w)))

        a = get_alpha_test(alpha_file)
        a_h, a_w = a.shape[:2]
        print('a_h, a_w: ' + str((a_h, a_w)))

        alpha = np.zeros((bg_h, bg_w), np.float32)
        alpha[0:a_h, 0:a_w] = a

        alpha = np.where(alpha < 50, 0, alpha)
        alpha = np.where(alpha > 200, 255, alpha)
        alpha = np.where(alpha % 255 == 0 , alpha, 255)
        # alpha = np.where((alpha > 100) & (alpha < 200), 128, alpha)

        kernel = np.ones((17, 17),np.uint8)
        alpha = cv.erode(alpha, kernel,iterations = 1)

        trimap = generate_trimap(alpha)
        # trimap = alpha

        # crop_size = random.choice(different_sizes)
        # x, y = random_choice(trimap, crop_size)
        x = 0 # start loc
        y = 0
        crop_size = (bg_h, bg_w)
        print('x, y: ' + str((x, y)))

        bgr_img_re = safe_crop(bgr_img, x, y, crop_size) # crop and resize
        alpha = safe_crop(alpha, x, y, crop_size)
        trimap = safe_crop(trimap, x, y, crop_size)
        # cv.imwrite('images/new_image.png', np.array(bgr_img).astype(np.uint8))
        # cv.imwrite('images/new_trimap.png', np.array(trimap).astype(np.uint8))
        # cv.imwrite('images/new_alpha.png', np.array(alpha).astype(np.uint8))

        x_test = np.empty((1, img_rows, img_cols, 4), dtype=np.float32)
        x_test[0, :, :, 0:3] = bgr_img_re / 255.
        x_test[0, :, :, 3] = trimap / 255.

        y_true = np.empty((1, img_rows, img_cols, 2), dtype=np.float32)
        y_true[0, :, :, 0] = alpha / 255.
        y_true[0, :, :, 1] = trimap / 255.

        y_pred = final.predict(x_test)
        # print('y_pred.shape: ' + str(y_pred.shape))

        y_pred = np.reshape(y_pred, (img_rows, img_cols))
        print(y_pred.shape)
        y_pred = y_pred * 255.0
        y_pred = get_final_output(y_pred, trimap)
        y_pred = y_pred.astype(np.uint8)

        sad_loss = compute_sad_loss(y_pred, alpha, trimap)
        mse_loss = compute_mse_loss(y_pred, alpha, trimap)
        str_msg = 'sad_loss: %.4f, mse_loss: %.4f, crop_size: %s' % (sad_loss, mse_loss, str(crop_size))
        print(str_msg)

        out = y_pred.copy()
        draw_str(out, (10, 20), str_msg)
        # cv.imwrite('images/new_out.png', out)

        bg = cv.imread(bg_file)
        # bh, bw = bg.shape[:2]
        # wratio = img_cols / bw
        # hratio = img_rows / bh
        # ratio = wratio if wratio > hratio else hratio
        # if ratio > 1:
        #    bg = cv.resize(src=bg, dsize=(math.ceil(bw * ratio), math.ceil(bh * ratio)), interpolation=cv.INTER_CUBIC)

        # bg = bg[0:bg_h, 0:bg_w]
        y_pred = cv.resize(src=y_pred, dsize=(bg_w, bg_h), interpolation=cv.INTER_CUBIC)

        im, bg = composite4(bgr_img, bg, y_pred, bg_w, bg_h)

                
        cv.imwrite('images/{:04d}.jpg'.format(idx), im)
        # assert 1==0
