import multiprocessing as mp
import os
import time
from functools import partial
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from stitch import Blend


# 辅助函数，用于去除离群点（基于中位数绝对偏差，MAD方法）
def remove_outliers(data, num_std=3):
    """
    使用中位数绝对偏差（MAD）方法去除离群点。

    参数：
    data (list or numpy array)：数据列表或数组。
    num_std (int)：判断离群点的标准差倍数，默认3倍标准差。

    返回：
    numpy array：去除离群点后的数据。
    """
    if len(data) < 2:
        return data
    median = np.median(data)
    mad = np.median(np.abs(data - median))
    if mad == 0:
        return data
    std_approx = 1.4826 * mad
    diff = np.abs(data - median)
    inlier_mask = diff < num_std * std_approx
    return data[inlier_mask]


def crop_by_features(img1,img2):
    # 转换为灰度图
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 创建ORB特征检测器
    orb = cv2.ORB_create()

    # 检测特征点并计算描述子
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    # 创建暴力匹配器
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # 进行特征匹配
    matches = bf.match(des1, des2)

    # 按照距离排序匹配结果
    matches = sorted(matches, key=lambda x: x.distance)

    # 选择部分较好的匹配点（这里选择前100个，可根据实际调整）
    good_matches = matches[:50]

    # 获取每对匹配特征点的横纵坐标
    x_coords_img1 = np.array([kp1[m.queryIdx].pt[0] for m in good_matches])
    y_coords_img1 = np.array([kp1[m.queryIdx].pt[1] for m in good_matches])
    x_coords_img2 = np.array([kp2[m.trainIdx].pt[0] for m in good_matches])
    y_coords_img2 = np.array([kp2[m.trainIdx].pt[1] for m in good_matches])

    # 计算纵坐标差值
    y_subtract_coords = abs(y_coords_img1 - y_coords_img2)

    # 去除纵坐标平均值中的离群点
    filtered_y_subtract_coords = remove_outliers(y_subtract_coords)

    # 根据过滤后的纵坐标平均值确定高度裁剪范围
    min_y_crop = int(np.mean(filtered_y_subtract_coords))

    # 同样的方式处理横坐标，计算横坐标平均值（每对匹配点）
    x_subtract_coords = abs(x_coords_img1 - x_coords_img2)

    # 去除横坐标平均值中的离群点
    filtered_x_subtract_coords = remove_outliers(x_subtract_coords)

    # 根据过滤后的横坐标平均值确定宽度裁剪范围
    min_x_crop = int(np.mean(filtered_x_subtract_coords))

    c1 = (0, 0)
    c2 = (min_x_crop, min_y_crop)

    return c1, c2


def crop_by_standard():
    c1 = (1646, 0)
    c2 = (0, 1302)
    return c1, c2


def imgCat(filename, phL, phR, phF, imgListR, crop1=False, crop2=False):
    imgR_path = os.path.join(phR, filename)
    current_index = imgListR.index(filename)
    imgR_next_name = imgListR[current_index + 1]
    imgL = cv2.imread(os.path.join(phL, filename))
    imgR = cv2.imread(imgR_path)
    imgR_next = cv2.imread(os.path.join(phR, imgR_next_name))
    imgR_new = cv2.vconcat([imgR,imgR_next])
    h = imgL.shape[0]
    w = imgL.shape[1]
    # crop1, crop2 = crop_by_standard()
    crop1, crop2 = crop_by_features(imgL, imgR)
    if crop1:
        imgL_crop = imgL[:, :w-crop1[0], :]
    if crop2:
        imgR_crop = imgR_new[crop2[1]:crop2[1]+h, crop2[0]:, :]
    imgStitch = cv2.hconcat([imgL_crop, imgR_crop])
    cv2.imwrite(os.path.join(phF,filename),imgStitch)


def multi_process(pL, pR, pF):
    if len(imglistL) > len(imglistR):
        files = imglistR
    else:
        files = imglistL
    # 显示进度条
    with mp.Pool(6) as pool:
        list(tqdm(pool.imap(partial(
            imgCat, phL=pL, phR=pR, phF=pF, imgListR=imglistR), files[:-1]), total=len(files)))
    pool.close()        # 关闭进程池，不再接受新的进程
    pool.join()         # 主进程阻塞等待子进程的退出


if __name__ == "__main__":
    t1 = time.time()
    mp.set_start_method('spawn')
    path = '/home/veily/1sjq/DataSet/RoadImg/ImgStitch/DR_1109_2'
    pathL = os.path.join(path, 'GroundImgL')
    pathR = os.path.join(path, 'GroundImgR')
    pathFinal = os.path.join('./', '2')
    Path(pathFinal+"/1").parent.mkdir(parents=True, exist_ok=True)
    imglistL = os.listdir(pathL)
    imglistR = os.listdir(pathR)
    imglistL.sort()
    imglistR.sort()
    multi_process(pathL, pathR, pathFinal)
    t2 = time.time()
    print('总耗时：', t2-t1)


