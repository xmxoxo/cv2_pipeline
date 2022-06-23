#!/usr/bin/env python3
#coding:utf-8

__author__ = 'xmxoxo<xmxoxo@qq.com>'

from cv2lib import *
#-----------------------------------------

# 读取照片
image = cv2.imread('1.png')
# 修改尺寸
image = cv2.resize(image, None, fx=0.5, fy=0.5)


# 流水处理
showpic(image, wname='orgian image')

#pipline(image, [gray, (mask,(78,255,1))], showimg=0, wname='gray_mask_18_255')
#pipline(image, [blur, gray, (mask, (18,255,1))], showimg=0, wname='blur_gray_mask_18_128')

image_bg = pipline(image, [blur, gray], showimg=0, wname='image_bg')
showpic(image_bg, wname='image_bg')

h,w,c = image.shape

# cut  可分成上下两个部分分别处理，然后再合并
image_u = image[:w//4]
image_d = image[w//4:]
# 背景 上半部分
image_bg_u = image_bg[:w//4] 
# 背景 下半部分
image_bg_d = image_bg[w//4:] 
#showpic(image_bg_u, wname='image_bg_u')

# 得到边缘线
#image_inv = pipline(image_u, [blur, gray, (mask, (28, 255, 0))], showimg=0, wname='up_blur_gray_mask_28')
#image_a = pipline(image_u, [blur, gray, (mask, (28, 255, 3))], showimg=0, wname='up_blur_gray_mask')

# 得到头发部分
image_n = pipline(image_u, [blur, gray, (mask, (28, 255, 1))], showimg=0, wname='up_blur_gray_mask_38')
showpic(image_n, wname='hair image')
#image_nu_gray = gray(image_n, showimg=1)

image_u_ret = image_bg_u.copy()
image_u_ret[image_n==255] = 255

# 合并
#image_u_ret = np.expand_dims(image_u_ret,2).repeat(3,axis=2)
print(image_u_ret.shape)
print(image_d.shape)
image_ret = np.vstack((image_u_ret, image_bg_d))

#showpic(image_u_ret, wname='image_u_ret')
#showpic(image_ret, wname='image_ret')

#image_ret = np.expand_dims(image_ret,2).repeat(3,axis=2)
image_cut_ret = pipline(image_ret, [gray, (mask, (128, 255, 0))], showimg=0, wname='image_cut_ret')

'''
# 腐蚀 膨胀
# MORPH_CLOSE, MORPH_OPEN
# referer:https://blog.csdn.net/yan_520csdn/article/details/101194165
op = cv2.MORPH_OPEN 进行开运算，指的是先进行腐蚀操作，再进行膨胀操作
op = cv2.MORPH_CLOSE 进行闭运算， 指的是先进行膨胀操作，再进行腐蚀操作
开运算：表示的是先进行腐蚀，再进行膨胀操作
闭运算：表示先进行膨胀操作，再进行腐蚀操作
'''
pp = 5
MaskImgEx = cv2.morphologyEx(image_cut_ret, cv2.MORPH_CLOSE, np.ones((pp, pp), np.uint8))
showpic(MaskImgEx, wname='MaskImgEx')

# 寻找最大的区域
contour, area = find_max_contour(MaskImgEx)

# 生成蒙板图
MaskImg3 = np.zeros_like(image.copy())
mask_person = cv2.drawContours(MaskImg3, [contour], -1, (255, 255, 255), cv2.FILLED)
mask_person = cv2.bitwise_not(mask_person)

showpic(mask_person, wname='mask_person')
print('mask_person:', mask_person.shape)

# 使用蒙板扣图
image_person = cv2.add(mask_person, image)

showpic(image_person, wname='person result')

cv2.imwrite(f'resut.png', image_person)
#-----------------------------------------
def old_code():
    #g = cv2.cvtColor(hsv, cv2.COLOR_BGR2GRAY)
    #showpic(g, wname='hsv_gray')
    #th, MaskImg = cv2.threshold(g, 128, 255, cv2.THRESH_BINARY_INV)
    #showpic(MaskImg, wname='hsv_MaskImg')

    # 图片的二值化黑白处理
    lower_black = np.array([0,0,0])
    # HSV 范围
    upper_black = np.array([180, 255, 30])
    heibai = cv2.inRange(hsv, lower_black, upper_black)

    # 闭运算
    k = np.ones((5, 5), np.uint8)
    r = cv2.morphologyEx(heibai, cv2.MORPH_CLOSE, k)
    # 颜色替换
    imageNew = np.copy(image)
    rows, cols, channels = image.shape

    for i in range(rows):
        for j in range(cols):
            # print("*")
            print(r[i, j])
            if r[i, j] == 255:
                # 像素点为255表示的是白色，我们就是要将白色处的像素点，替换为红色
                imageNew[i, j] = (255, 255, 255)
                # 此处替换颜色，为BGR通道，不是RGB通道
    # 显示
    cv2.imshow('image', image)
    cv2.imshow('hsv', hsv)
    cv2.imshow('heibai', heibai)
    cv2.imshow('r', r)
    cv2.imshow('imageNew', imageNew)


# 无限等待
cv2.waitKey(0)
# 销毁内存
cv2.destroyAllWindows()


if __name__ == '__main__':
    pass

