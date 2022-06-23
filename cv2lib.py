#!/usr/bin/env python3
#coding:utf-8

__author__ = 'xmxoxo<xmxoxo@qq.com>'

import cv2
import numpy as np

def loadpic(fname):
    image = cv2.imread(fname)
    return image

def resize(image, parm=(0.5, 0.5), showimg=0):
    image = cv2.resize(image, None, fx=parm[0], fy=parm[1])
    if showimg:
        showpic(image, wname='Resize')
    return image

def showpic(image, wname='image'):
    #if image.ndim==2:
    #    image = np.expand_dims(image,2).repeat(3,axis=2)
    cv2.imshow(wname, image)
    #cv2.waitKey(0)

def hsv(image, showimg=0):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    if showimg:
        showpic(hsv, wname='hsv')
    return hsv

# 锐化
def blur(image, parm=8, showimg=0):
    kernel = np.array([[ 0, -1,  0],
                       [-1, parm, -1],
                       [ 0, -1,  0]], np.float32) #锐化
    image = cv2.filter2D(image, -1, kernel=kernel)
    if showimg:
        showpic(image, wname='BLUR')
    return image

# 转成灰度
def gray(image, showimg=0):
    LightImgGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if showimg:
        showpic(LightImgGray, wname='GRAY')
    return LightImgGray

# 反色
def bitnot(image, showimg=0):
    Img = cv2.bitwise_not(image)
    if showimg:
        showpic(Img, wname='BITNOT')
    return Img


# 开运算
def op_open(image, parm=(5,5), showimg=0):
    k = np.ones(parm, np.uint8)
    ImgRet = cv2.erode(image, k)
    ImgRet = cv2.dilate(image, k)
    if showimg:
        showpic(ImgRet, wname='OPEN')
    return ImgRet

# 闭运算
def op_close(image, parm=(5,5), showimg=0):
    k = np.ones(parm, np.uint8)
    ImgRet = cv2.dilate(image, k)
    ImgRet = cv2.erode(image, k)
    if showimg:
        showpic(ImgRet, wname='CLOSE')
    return ImgRet


# 提取蒙板二值化
'''
threshold 二值化
阈值	小于阈值	大于阈值
  0 	置0(黑) 	置填充色        THRESH_BINARY
  1  	置填充色	置0(黑)     THRESH_BINARY_INV
  2 	保持原色	置灰色
  3 	置0(黑)     保持原色        THRESH_TOZERO
  4 	保持原色	置0(黑)     THRESH_TOZERO_INV

blur后有三段颜色分布：黑<-- 0-30头发 ， 30-128 背景, 128-255身  -->白
'''
def mask(image, parm=(28,255,1), showimg=0):
    th, MaskImg = cv2.threshold(image, parm[0], parm[1], parm[2]) #, cv2.THRESH_BINARY_INV)
    if showimg:
        showpic(MaskImg, wname='MaskImg')
    #if MaskImg.ndim==2:
    #    MaskImg = np.expand_dims(MaskImg,2).repeat(3,axis=2)
    return MaskImg

# 归一化
def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

# 流水处理, showimg:是否显示过程图，showresult表示显示结果图
def pipline(image, oplist, showimg=0, showresult=0, wname='result'):
    #static idx
    nimage = image.copy()
    if nimage.ndim==2:
        nimage = np.expand_dims(nimage,2).repeat(3,axis=2)
    if showimg:
        #idx += 1
        showpic(image, wname='image')

    for op in oplist:
        if type(op)==tuple:
            fun, parm = op
            nimage = fun(nimage, parm, showimg=showimg)
        else:
            nimage = op(nimage, showimg=showimg)
    if showresult:
        showpic(nimage, wname=wname)
    return nimage

def find_max_contour(MaskImgEx):
    '''寻找一个二值图像的轮廓，黑底 找白色
    cv2.RETR_EXTERNAL=0 外部轮廓用的比较多
    cv2.RETR_LIST = 1
    cv2.RETR_CCOMP = 2, 
    cv2.RETR_TREE = 3, 
    cv2.RETR_FLOODFILL = 4
    referer:https://blog.csdn.net/weixin_42216109/article/details/89840323
    第三个参数：ContourApproximationModes
      cv2.CHAIN_APPROX_NONE = 1, 轮廓所有点
      cv2.CHAIN_APPROX_SIMPLE = 2, 只返回四角的点 <--默认
      cv2.CHAIN_APPROX_TC89_L1= 3, 
      cv2.CHAIN_APPROX_TC89_KCOS= 4
    https://blog.csdn.net/qq_15642411/article/details/79930083
    返回结果：contour, max_area
    '''
    contours, hierarchy = cv2.findContours(MaskImgEx, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contour, max_area = None,None
    if len(contours) >0 :
        #print('contours:', len(contours))        
        # 计算各块的面积
        area_list = [cv2.contourArea(x) for x in contours]
        #得到最大的一块面积
        max_area_index = np.argmax(area_list) 
        contour = contours[max_area_index]
        max_area = area_list[max_area_index]

    return contour, max_area

def find_contours(MaskImgEx):
    '''寻找一个二值图像的轮廓，黑底 找白色
    第二个参数：
    cv2.RETR_EXTERNAL=0 外部轮廓用的比较多
    cv2.RETR_LIST = 1
    cv2.RETR_CCOMP = 2, 
    cv2.RETR_TREE = 3, 
    cv2.RETR_FLOODFILL = 4
    referer:https://blog.csdn.net/weixin_42216109/article/details/89840323
    第三个参数：ContourApproximationModes
      cv2.CHAIN_APPROX_NONE = 1, 轮廓所有点
      cv2.CHAIN_APPROX_SIMPLE = 2, 只返回四角的点 <--默认
      cv2.CHAIN_APPROX_TC89_L1= 3, 
      cv2.CHAIN_APPROX_TC89_KCOS= 4
    https://blog.csdn.net/qq_15642411/article/details/79930083
    
    返回结果：contour, max_area
    '''
    contours, hierarchy = cv2.findContours(MaskImgEx, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    '''
    contour, max_area = None,None
    if len(contours) >0 :
        #print('contours:', len(contours))        
        # 计算各块的面积
        area_list = [cv2.contourArea(x) for x in contours]
        #得到最大的一块面积
        max_area_index = np.argmax(area_list) 
        contour = contours[max_area_index]
        max_area = area_list[max_area_index]
    '''

    return contours, hierarchy

# 返回最大区域块的蒙板, 注意返回结果只有1个通道 
def get_max_area(img, show=0, wname='mask'):
    contour, _ = find_max_contour(img)
    mask = cv2.drawContours(np.zeros_like(img.copy()), [contour], -1, (255, 255, 255), cv2.FILLED)
    if show:
        showpic(mask, wname=wname)
    return mask, contour

def wait(exit=0):
    # 无限等待
    cv2.waitKey(0)
    # 销毁内存
    cv2.destroyAllWindows()
    if exit==1:
        sys.exit()

    
def draw_rectange(img, contours):
    pass

    
def draw_contours(image, contours, color=(0,0,255), width=1, showimg='Image_box'):
    ''' 画出区域
    '''

    print('找到%d个区域' % len(contours))
    for i in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[i])
        print('第%d个区块，位置:(%d, %d) 大小：%d x %d ,面积:%d ' % (i, x,y, w, h, 0))

        cv2.rectangle(image, (x,y), (x+w,y+h), color, width)
    if showimg:
        showpic(image, wname=showimg)



if __name__ == '__main__':
    pass

