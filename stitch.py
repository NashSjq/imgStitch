import copy
import os
import time

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from torchvision import transforms

from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image, rbd

torch.set_grad_enabled(False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# SuperPoint+LightGlue
extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)
matcher = LightGlue(features='superpoint', depth_confidence=0.9,
                    width_confidence=0.95).eval().to(device)


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


class FeatureMatch:
    def __init__(self, imgL, imgR):
        self.imgL = imgL
        self.imgR = imgR
        
    def sift_cv(self):
        # 匹配特征点阈值
        MIN = 10
        ts = time.time()
        # 创建SIFT特征点检测
        sift = cv2.SIFT_create()
        # 检测兴趣点并计算描述子, 
        kp1, describe1 = sift.detectAndCompute(self.imgL, None)
        kp2, describe2 = sift.detectAndCompute(self.imgR, None)
        te = time.time()
        print('兴趣点检测耗时:',te-ts)
        ts = time.time()
        # 使用OpenCV中的FLANN匹配算法进行特征匹配，并返回最近邻和次近邻匹配的结果
        FLANN_INDEX_KDTREE = 0
        indexParams = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        searchParams = dict(checks=50)
        flann = cv2.FlannBasedMatcher(indexParams,searchParams)
        matches = flann.knnMatch(describe1, describe2, k=2)
        te = time.time()
        print('特征匹配耗时:',te-ts)
        ts = time.time()
        # 储存特征匹配最好的优质匹配点对
        '''基于距离阈值选择优质匹配点对，如果最近邻m的距离小于0.65倍的次近邻n的距离，
        则认为这个匹配点对是优质的，将它存储在good列表中。'''
        good = []
        for m,n in matches:
            if m.distance < 0.65 * n.distance:
                good.append(m)
        te=time.time()
        print('储存最优匹配点耗时：', te-ts)
        if len(good) > MIN:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            tge_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            return src_pts,tge_pts
        else:
            print("not enough matches!")
            return 0,0

  
    def light_glue(self):
        # load tensor img
        image0 = self.imgL if torch.is_tensor(self.imgL) else transforms.ToTensor()(self.imgL).to(device)
        image1 = self.imgR if torch.is_tensor(self.imgR) else transforms.ToTensor()(self.imgR).to(device)
        # extract local features
        feats0 = extractor.extract(image0)  # auto-resize the image, disable with resize=None
        feats1 = extractor.extract(image1)
        # match the features
        matches01 = matcher({'image0': feats0, 'image1': feats1})
        # remove batch dimension
        feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]  
        # indices with shape (K,2)
        matches = matches01['matches']  
        # coordinates in image #0, shape (K,2)
        points0 = feats0['keypoints'][matches[..., 0]]  
        # coordinates in image #1, shape (K,2)
        points1 = feats1['keypoints'][matches[..., 1]] 
        p1,p2 = np.array(points0.cpu()), np.array(points1.cpu())
        return p1,p2


class Align:
    def __init__(self, points1, points2, img, img_path, filename):
        self.points1 = points1
        self.points2 = points2
        self.img =img
        self.img_path = img_path
        self.filename = filename

    def averge_distance(self):
        imgListR = os.listdir(self.img_path)
        imgListR.sort()
        current_index = imgListR.index(self.filename)
        imgR_next_name = imgListR[current_index + 1]
        imgR_next = cv2.imread(os.path.join(self.img_path, imgR_next_name))
        imgR_new = cv2.vconcat([self.img, imgR_next])
        h = self.img.shape[0]

        x_coords_img1, y_coords_img1 = self.points1[:, 0], self.points1[:, 1]
        x_coords_img2, y_coords_img2 = self.points2[:, 0], self.points2[:, 1]
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
        c2 = (min_x_crop, min_y_crop)
        imgR_crop = imgR_new[c2[1]:c2[1]+h, :, :]

        return imgR_new



class Blend:
    def __init__(self, imgL, warpimg):
        self.imgL = imgL
        # self.imgL_cuda = imgL_cuda
        self.warpimg = warpimg
        self.rows, self.cols = self.imgL.shape[:2]
        self.left, self.right = self._getOverlopEdge()
        self.threshold=20
        self.overlopLen=200
        self.max_leveln = self._getMaxMiltBland()


    def time_of_function(func):
        def inner(s):
            ts = time.time()
            k = func(s)
            te = time.time()
            print(f'{func.__name__} time: {te-ts}s')
            return k
        return inner


    def _getOverlopEdge(self):
        left = 0
        right = self.cols
        # 找到img1和warpimg重叠的最左边界
        for col in range(0, self.cols):
            if self.imgL[:, col].any() and self.warpimg[:, col].any():
                left = col
            break
        # # 找到img1和warpimg重叠的最右边界（以左图为基准变换，右边界固定）
        # for col in range(self.cols - 1, 0, -1):
        #     if self.imgL[:, col].any() and self.warpimg[:, col].any():
        #         right = col
        #     break
        return left, right


    def _getEdgeCol(self, img):
        for i in range(img.shape[1]):
            if img[:, -(i+1)].max() > 0:
                return i


    def _getMaxMiltBland(self):
        max_levelnn = int(np.floor(np.log2(min(self.imgL.shape[0], self.imgL.shape[1],
                                          self.warpimg.shape[0], self.warpimg.shape[1]))))
        return max_levelnn


    def LaplacianPyramid(self, img, leveln):
        LP = []
        for i in range(leveln - 1):
            next_img = cv2.pyrDown(img)
            res_img = cv2.subtract(img, cv2.pyrUp(next_img, dstsize=img.shape[1::-1]))
            LP.append(res_img)
            img = next_img
        LP.append(img)
        return LP
    
    
    @time_of_function
    def weightAverage(self):
        ''' overlapping area weight-average blending
            replacing by weightAverage_martix
        '''
        res = np.zeros([self.rows, self.cols, 3], np.uint8)

        for row in range(0, self.rows):
            for col in range(0, self.cols):
                # 左图的像素点为0，则添加右图此点
                if not self.imgL[row, col].any():
                    res[row, col] = self.warpimg[row, col]
                # 右图的像素点为0，则添加左图此点
                elif not self.warpimg[row, col].any():
                    res[row, col] = self.imgL[row, col]
                else:
                    # 重叠部分加权平均
                    srcimgLen = float(abs(col - self.left))
                    testimgLen = float(abs(col - self.right))
                    alpha = srcimgLen / (srcimgLen + testimgLen)
                    res[row, col] = np.clip(self.imgL[row, col] * (1 - alpha) + self.warpimg[row, col] * alpha, 0, 255)
        img_blend = self.warpimg
        img_blend[0:self.rows, 0:self.cols] = res
        return img_blend


    # @time_of_function
    def weightAverage_matrix(self):
        ''' overlapping area weight-average blending 
        '''
        pano = copy.deepcopy(self.warpimg)
        pano[0:self.rows, 0:self.cols] = self.imgL

        rows = pano.shape[0]
        overlopL = self.right - self.overlopLen
        # calculate weight matrix
        alphas = np.array([self.right - np.arange(overlopL, self.right)] * rows) / (self.right - overlopL)
        alpha_matrix = np.ones((alphas.shape[0], alphas.shape[1], 3))
        alpha_matrix[:, :, 0] = alphas
        alpha_matrix[:, :, 1] = alphas
        alpha_matrix[:, :, 2] = alphas
        # common area one image no pixels
        alpha_matrix[self.warpimg[0:rows, overlopL:self.right, :] <= self.threshold] = 1

        img_tar = pano[:, 0:self.cols]
        pano[0:rows, overlopL:self.right] = img_tar[0:rows, overlopL:self.right] * alpha_matrix \
                                       + self.warpimg[0:rows, overlopL:self.right] * (1 - alpha_matrix)

        return pano


    @time_of_function
    def linerMix(self):
        '''diect blend by adding pixes, with cpu
        '''
        pano = copy.deepcopy(self.warpimg)
        pano[0:self.rows, 0:self.cols] = self.imgL

        rows = pano.shape[0]
        overlopL = self.cols - self.overlopLen
        # calculate weight matrix
        alphas = np.array([np.ones(self.overlopLen)*1.8] * rows)
        alpha_matrix = np.ones((alphas.shape[0], alphas.shape[1], 3))
        alpha_matrix[:, :, 0] = alphas
        alpha_matrix[:, :, 1] = alphas
        alpha_matrix[:, :, 2] = alphas
        # common area one image no pixels
        alpha_matrix[self.warpimg[0:rows, overlopL:self.right, :] <= self.threshold] = 1

        img_tar = pano[:, 0:self.cols]
        pano[0:rows, overlopL:self.right] = (img_tar[0:rows, overlopL:self.right] \
                                             * alpha_matrix + self.warpimg[0:rows, overlopL:self.right]) / alpha_matrix

        return pano


    @time_of_function
    def linerMix_cuda(self):
        '''diect blend by adding pixes, 
           using cuda accelerate, but speed longger than cpu
           replace by  linerMix
        '''
        warpimg_cuda = transforms.ToTensor()(self.warpimg).cuda()
        pano = copy.deepcopy(warpimg_cuda)
        # pano[:, 0:self.rows, 0:self.cols] = self.imgL_cuda

        rows = pano.shape[1]
        overlopL = self.cols - self.overlopLen

        alpha1 = torch.ones(self.overlopLen).cuda()*1.8
        alpha2 = torch.cat([alpha1.unsqueeze(0)]*rows, dim=0)
        alpha3 = torch.cat([alpha2.unsqueeze(0)]*3, dim=0)

        alpha3[warpimg_cuda[:, 0:rows, overlopL:self.right] <= self.threshold/255] = 1

        # pano[:, :, overlopL:self.right] = (self.imgL_cuda[:, :rows, overlopL:self.right] \
        #                                 + warpimg_cuda[:, :rows, overlopL:self.right])/alpha3

        p = pano * 255
        pano_cpu = np.transpose(p[:, :self.rows, :].cpu().numpy().astype(np.uint8), (1, 2, 0))

        return pano_cpu


    @time_of_function
    def muilt_bland(self, leveln = 6):
        '''Multi-band Pyramid Fusion
        '''

        pano = copy.deepcopy(self.warpimg)
        pano[0:self.rows, 0:self.cols] = self.imgL
        
        if leveln is None:
            leveln = self.max_leveln
        if leveln < 1 or leveln > self.max_leveln:
            print ("warning: inappropriate number of leveln")
            leveln = self.max_leveln

        img1 = self.imgL[:, -self.overlopLen:]
        img2 = self.warpimg[:self.rows, self.right-self.overlopLen:self.right]

        # [G`5, G`4-G`5^1, G`3-G`4^1, G`2-G`3^1, G`1-G`2^1, G-G`1^1]
        lp1 = self.LaplacianPyramid(img1, leveln)
        lp2 = self.LaplacianPyramid(img2, leveln)
        
        # 将两张图片的拉普拉斯金字塔进行拼接
        LS = []
        for l1, l2 in zip(lp1, lp2):
            rows, cols, dpt = l1.shape
            ls = np.hstack((l1[:, 0:int(cols/2)], l2[:, int(cols/2):])) # horizontal
            LS.append(ls)
        
        # 重建图像
        ls_ = LS[-1]
        for i in range(1, leveln):
            ls_ = cv2.pyrUp(ls_, dstsize=LS[-i-1].shape[1::-1])
            ls_ = cv2.add(ls_, LS[-i-1]) # +

        pano[:self.rows, self.right-self.overlopLen:self.right] = ls_

        return pano
    

    def findOverlop(self):
        im3 = cv2.threshold(cv2.cvtColor(self.imgL, cv2.COLOR_BGR2GRAY),
                            0, 1, cv2.THRESH_BINARY)[1]
        for i in range(im3.shape[1]):
            if im3[:, i].max() > 0:
                break
        return self.imgL[:, i:], self.warpimg[:self.imgL.shape[0], i:self.imgL.shape[1]]


    def pix_Strength_value(self, I1, I2):
        I1 = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)
        I2 = cv2.cvtColor(I2, cv2.COLOR_BGR2GRAY)
        Sx = np.array([[-2, 0, 2], [-1, 0, 1], [-2, 0, 2]])
        Sy = np.array([[-2, -1, -2], [0, 0, 0], [2, 1, 2]])

        I1_Sx = cv2.filter2D(I1, -1, Sx)
        I1_Sy = cv2.filter2D(I1, -1, Sy)
        I2_Sx = cv2.filter2D(I2, -1, Sx)
        I2_Sy = cv2.filter2D(I2, -1, Sy)

        E_color = (I1 - I2) ** 2
        E_geometry = (I1_Sx - I2_Sx) * (I1_Sy - I2_Sy)
        E = E_color + E_geometry
        return E.astype(float)


    def optimal_seam_rule(self):
        i1, i2 = self.findOverlop()
        E = self.pix_Strength_value(i1,i2)
        # optimal seam
        paths_weight = E[0, 1:-1].reshape(1, -1)  # Cumulative strength value
        paths = np.arange(1, E.shape[1] - 1).reshape(1, -1)  # save index
        for i in range(1, E.shape[0]):
            # boundary process
            lefts_index = paths[-1, :] - 1
            lefts_index[lefts_index < 0] = 0
            rights_index = paths[-1, :] + 1
            rights_index[rights_index > E.shape[1] - 1] = E.shape[1] - 1
            mids_index = paths[-1, :]
            mids_index[mids_index < 0] = 0
            mids_index[mids_index > E.shape[1] - 1] = E.shape[1] - 1

            # compute next row strength value(remove begin and end point)
            lefts = E[i, lefts_index] + paths_weight[-1, :]
            mids = E[i, paths[-1, :]] + paths_weight[-1, :]
            rights = E[i, rights_index] + paths_weight[-1, :]
            # return the index of min strength value
            values_3direct = np.vstack((lefts, mids, rights))
            index_args = np.argmin(values_3direct, axis=0) - 1  #
            # next min strength value and index
            weights = np.min(values_3direct, axis=0)
            path_row = paths[-1, :] + index_args
            paths_weight = np.insert(
                paths_weight, paths_weight.shape[0], values=weights, axis=0)
            paths = np.insert(paths, paths.shape[0], values=path_row, axis=0)

        # search min path
        min_index = np.argmin(paths_weight[-1, :])
        return paths[:, min_index]


    def plot_seam(self):
        ''' plot the seam line'''
        len_overlop = self.right - self.left
        minSeam = self.optimal_seam_rule()
        fig, ax = plt.subplots(figsize=(14, 7))
        fig.show(self.imgL)
        a = []
        for i in range(self.rows):
            a.append(i)
        ax.plot(self.cols - len_overlop+minSeam, a)


    def seam_blend(self):
        minSeam = self.optimal_seam_rule()
        len_overlop = self.right - self.left
        for i in range(self.rows):
            self.warpimg[i, :(self.cols-len_overlop+minSeam[i])
                    ] = self.imgL[i, :(self.cols-len_overlop+minSeam[i])]
        # num = imgcrop_cuda(warpimg[:imgL.shape[0], :])
        return self.warpimg[:self.rows, :]
    

class Crop:
    def __init__(self, img, rows):
        self.img = img
        self.rows, self.cols = self.img.shape[:2]
        self.row_crop = rows


    def maximum_internal_rectangle(img):
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        contour = contours.reshape(len(contours[0]), 2)
    
        rect = []
    
        for i in range(len(contour)):
            x1, y1 = contour[i]
            for j in range(len(contour)):
                x2, y2 = contour[j]
                area = abs(y2 - y1) * abs(x2 - x1)
                rect.append(((x1, y1), (x2, y2), area))
    
        all_rect = sorted(rect, key=lambda x: x[2], reverse=True)
    
        if all_rect:
            best_rect_found = False
            index_rect = 0
            nb_rect = len(all_rect)
    
            while not best_rect_found and index_rect < nb_rect:
    
                rect = all_rect[index_rect]
                (x1, y1) = rect[0]
                (x2, y2) = rect[1]
    
                valid_rect = True
    
                x = min(x1, x2)
                while x < max(x1, x2) + 1 and valid_rect:
                    if img[y1, x] == 0 or img[y2, x] == 0:
                        valid_rect = False
                    x += 1
    
                y = min(y1, y2)
                while y < max(y1, y2) + 1 and valid_rect:
                    if img[y, x1] == 0 or img[y, x2] == 0:
                        valid_rect = False
                    y += 1
    
                if valid_rect:
                    best_rect_found = True
    
                index_rect += 1
    
            if best_rect_found:
                # 如果要在灰度图img_gray上画矩形，请用黑色画（0,0,0）
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 1)
                cv2.namedWindow("enhanced",0)
                cv2.resizeWindow("enhanced", 640, 480)
                cv2.imshow("enhanced", img)
                cv2.waitKey(0)
    
            else:
                print("No rectangle fitting into the area")
    
        else:
            print("No rectangle found")

    def erodeImg(img):
        # 最小外接矩形
        mask = np.zeros(img.shape, dtype="uint8")
        (x, y, w, h) = cv2.boundingRect(img)  
        cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
        minRect = mask.copy()
        sub = mask.copy()
        # 腐蚀处理，缺点：不是最大内接矩形
        while cv2.countNonZero(sub) > 0:
            minRect = cv2.erode(minRect, None)
            # sub判断minRect中白色的部分是否小于thresh
            sub = cv2.subtract(minRect, img)
        return minRect

    def imgcrop(img,mask):
        '''图像裁剪'''
        # 图像边缘扩展黑边，方便后续寻找轮廓
        stitched = cv2.copyMakeBorder(img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, (0, 0, 0))
        # 转单通道
        gray = cv2.cvtColor(stitched, cv2.COLOR_BGR2GRAY)
        # 二值化处理
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]
        for i  in range(thresh.shape[1]):
            if cv2.countNonZero(thresh[:,-i]) > 0:
                thresh = thresh[:,:-i]
                break

        # # 轮廓顶点，长宽
        (x, y, w, h) = cv2.boundingRect(thresh)
        # # 裁剪
        stitched = stitched[y:y + h, x:x + w]
        return stitched
    
    def single_edge_crop(self):
        for i in range(self.cols):
            if self.img[:, -(i+1)].max() > 0:
                break
        return self.img[:self.row_crop, :-i-1]
