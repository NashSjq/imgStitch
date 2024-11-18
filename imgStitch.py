import multiprocessing as mp
import os
import time
from functools import partial
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from stitch import Align, Blend, Crop, FeatureMatch


# Main
def imgStitch(filename, phL, phR, phF):
    try:
        # 读取待拼接图像
        imgL = cv2.imread(os.path.join(phL, filename))
        imgR = cv2.imread(os.path.join(phR, filename))
        # LightGlue匹配特征点
        featureMatch = FeatureMatch(imgL, imgR)
        src_pts, tge_pts = featureMatch.light_glue()
        # 左右图像对齐
        aligning = Align(src_pts, tge_pts, imgR, phR, filename)
        imgR = aligning.averge_distance()
        # RANSAC算法计算单应性矩阵
        M, mask = cv2.findHomography(src_pts, tge_pts, cv2.RANSAC, 2)
        # imgR图像透视变换
        warpimg = cv2.warpPerspective(imgR, np.linalg.inv(
            M), (imgL.shape[1] + imgR.shape[1], imgR.shape[0]))
        # 融合
        blending = Blend(imgL, warpimg)
        # # 加权融合
        img_blend = blending.weightAverage_matrix()
        # 裁剪
        croping = Crop(img_blend, imgL.shape[0])
        img_crop = croping.single_edge_crop()
        # save
        cv2.imwrite(os.path.join(phF, filename), img_crop)
    except:
        print(f'imgStitch failed: {filename}')


def multi_process(pL, pR, pF):
    filesL = os.listdir(pL)
    filesR = os.listdir(pR)
    if len(filesL) > len(filesR):
        files = filesR
    else:
        files = filesL
    files = files[:-1]

    # 显示进度条
    with mp.Pool(4) as pool:
        list(tqdm(pool.imap(partial(
            imgStitch, phL=pL, phR=pR, phF=pF), files), total=len(files)))
    pool.close()        # 关闭进程池，不再接受新的进程
    pool.join()         # 主进程阻塞等待子进程的退出


if __name__ == "__main__":
    t1 = time.time()
    mp.set_start_method('spawn')
    path = '/home/veily/1sjq/DataSet/RoadImg/ImgStitch/DR_1109_1'
    pathL = os.path.join(path, 'GroundImgL')
    pathR = os.path.join(path, 'GroundImgR')
    pathFinal = os.path.join('../../DataSet/', '241114_1')
    Path(pathFinal+"/1").parent.mkdir(parents=True, exist_ok=True)
    multi_process(pathL, pathR, pathFinal)
    t2 = time.time()
    print('总耗时：', t2-t1)
