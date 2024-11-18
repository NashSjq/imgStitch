import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from stitch import Align, Blend, Crop, FeatureMatch

torch.set_grad_enabled(False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def mirror_img(imgPath, savePath):
    Path(savePath).parent.mkdir(parents=True, exist_ok=True)
    im = cv2.imread(imgPath)
    img = cv2.flip(im, 1)
    cv2.imwrite(savePath, img)


def mirrorImgs():
    imgL_path = '../../DataSet/RoadImg/ImgStitch/DR_1109_2/GroundImgL/'
    for f in tqdm(os.listdir(imgL_path)):
        img_path = os.path.join(imgL_path, f)
        mirror_img(img_path, f'../../DataSet/RoadImg/ImgStitch/DR_1109_2/GroundImgL1/{f}')


def cat2img(im1_path, im2_path, paF, cat_way):
    img1 = cv2.imread(im1_path)
    img2 = cv2.imread(im2_path)
    if cat_way == 'v':
        imgStitch = cv2.vconcat([img1, img2])
    elif cat_way == 'h':
        imgStitch = cv2.hconcat([img1, img2])
    cv2.imwrite(paF,imgStitch)


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


def orb_match(img1, img2):
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
    return x_coords_img1, y_coords_img1, x_coords_img2, y_coords_img2


def crop_by_features(img1,img2,im3,h):

    # orb特征匹配获取特征点坐标
    # x_coords_img1, y_coords_img1, x_coords_img2, y_coords_img2 = orb_match(img1, img2)
    # lightglue获取特征点坐标

    # lightglue获取特征点坐标
    featureMatch = FeatureMatch(img1, img2)
    p1, p2 = featureMatch.light_glue()
    x_coords_img1, y_coords_img1 = p1[:, 0], p1[:, 1]
    x_coords_img2, y_coords_img2 = p2[:, 0], p2[:, 1]

    # 计算纵坐标差值
    y_subtract_coords = abs(y_coords_img1 - y_coords_img2)
    # 去除纵坐标平均值中的离群点
    filtered_y_subtract_coords = remove_outliers(y_subtract_coords)
    # 根据过滤后的纵坐标平均值确定高度裁剪范围
    min_y_crop = int(np.mean(filtered_y_subtract_coords))
    cropped_img2_height = im3[min_y_crop:min_y_crop+h, :, :]

    # 同样的方式处理横坐标，计算横坐标平均值（每对匹配点）
    x_subtract_coords = abs(x_coords_img1 - x_coords_img2)
    # 去除横坐标平均值中的离群点
    filtered_x_subtract_coords = remove_outliers(x_subtract_coords)
    # 根据过滤后的横坐标平均值确定宽度裁剪范围
    min_x_crop = int(np.mean(filtered_x_subtract_coords))
    cropped_img2 = cropped_img2_height[:, min_x_crop:, :]

    # 进行拼接，这里简单地水平拼接
    result = cv2.hconcat([img1, cropped_img2])

    # 显示拼接后的结果
    cv2.namedWindow('custom window', cv2.WINDOW_KEEPRATIO)
    cv2.imshow('custom window', result)
    cv2.resizeWindow('custom window', 200, 200)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def imgCat(filename, phL, phR, phF, imgListR, crop1=False, crop2=False):
    imgR_path = os.path.join(phR, filename)
    current_index = imgListR.index(filename)
    imgR_next_name = imgListR[current_index + 1]
    imgL = cv2.imread(os.path.join(phL, filename))
    imgR = cv2.imread(imgR_path)
    imgR_next = cv2.imread(os.path.join(phR, imgR_next_name))
    imgR_new = cv2.vconcat([imgR,imgR_next])
    h = imgL.shape[0]
    crop_by_features(imgL, imgR, imgR_new, h)
    if crop1:
        imgL_crop = imgL[:, :-crop1[0], :]
    if crop2:
        imgR_crop = imgR_new[crop2[1]:crop2[1]+h, crop2[0]:, :]
    imgStitch = cv2.hconcat([imgL_crop, imgR_crop])
    cv2.imwrite(os.path.join(phF,filename),imgStitch)


# Main
def imgStitch(phL, phR, phF, filename):
    imgL = cv2.imread(os.path.join(phL, filename))
    imgR = cv2.imread(os.path.join(phR, filename))
    # matching features
    featureMatch = FeatureMatch(imgL,imgR)
    src_pts, tge_pts = featureMatch.light_glue()
    # align img
    aligning = Align(src_pts, tge_pts, imgR, phR, filename)
    imgR = aligning.averge_distance()
    # homography
    M, mask = cv2.findHomography(src_pts, tge_pts, cv2.RANSAC, 2)
    warpimg = cv2.warpPerspective(imgR, np.linalg.inv(
        M), (imgL.shape[1] + imgR.shape[1], imgR.shape[0]))
    # # blending img
    blending = Blend(imgL, warpimg)
    '''linerMix_cuda time: 0.6562411785125732s'''
    # img_blend = blending.linerMix_cuda()
    '''linerMix time: 0.03527259826660156s'''
    # img_blend = blending.linerMix()
    '''muilt_bland time: 0.029062986373901367s'''
    # img_blend = blending.muilt_bland()
    '''weightAverage time: 61.04945468902588s'''
    # img_crop = blending.weightAverage()
    '''weightAverage_matrix time: 0.04328751564025879s'''
    img_blend = blending.weightAverage_matrix()
    # img_blend = blending.seam_blend()
    # 裁剪
    croping = Crop(img_blend, imgL.shape[0])
    img_crop = croping.single_edge_crop()
    cv2.imwrite(phF, img_crop)


if __name__ == "__main__":
    t1 = time.time()
    imPathL = '/home/veily/1sjq/DataSet/RoadImg/ImgStitch/DR_1109_1/GroundImgL'
    imPathR = '/home/veily/1sjq/DataSet/RoadImg/ImgStitch/DR_1109_1/GroundImgR'
    name = 'K000+196.jpg'
    pathFinal = './result.jpg'
    imgStitch(imPathL, imPathR, pathFinal, name)
    t2 = time.time()
    print('总耗时：', t2-t1)

    # cat2img('/home/veily/1sjq/DataSet/RoadImg/ImgStitch/DR_1109_1/GroundImgR/K000+003.jpg',
    #         '/home/veily/1sjq/DataSet/RoadImg/ImgStitch/DR_1109_1/GroundImgR/K000+004.jpg',
    #         './03.jpg', 'v')

    # path = '/home/veily/1sjq/DataSet/RoadImg/ImgStitch/DR_1109_1'
    # pathL = os.path.join(path, 'GroundImgL')
    # pathR = os.path.join(path, 'GroundImgR')
    # pathFinal = './'
    # Path(pathFinal+"/1").parent.mkdir(parents=True, exist_ok=True)
    # imglistR = os.listdir(pathR)
    # imglistR.sort()
    # c1 = (1646, 0)
    # c2 = (0, 1302)
    # imgCat('K000+196.jpg', pathL, pathR,  pathFinal, imglistR)

    
