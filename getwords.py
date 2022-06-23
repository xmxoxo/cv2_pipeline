#!/usr/bin/env python3
#coding:utf-8

__author__ = 'xmxoxo<xmxoxo@qq.com>'

'''
快速提取文本位置
'''

from cv2lib import *

# 读取照片
image = cv2.imread('word.png')
# 缩小一半
image = pipline(image, [(resize, (0.5,0.5))], showresult=1, wname='orgian image')
# pipline流水处理
MaskImgEx = pipline(image, [(blur, 7), 
                            gray,
                            #(op_open, (1, 1)), #(op_close, (1, 1)),
                            (op_open, (1, 1)), (op_close, (1, 1)), 
                            (op_close, (2, 3)),(op_close, (2, 3)),
                            (mask, (158, 255, 1))
                            ],
                            showimg=0, showresult=1, wname='MaskImgEx')

# 寻找区域
contours, hierarchy = find_contours(MaskImgEx)
# 画出区域
draw_contours(image, contours, color=(0,0,255), width=1, showimg='Result')

# 等待键盘
wait()


if __name__ == '__main__':
    pass

