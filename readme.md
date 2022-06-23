# 基于CV2的图像流式处理

思路 ：

对图像的各类处理，基本上都是对图像进行一系列的不同处理，把这些处理操作进行分支、叠加等，最终得到某些结果；

所以可以先将基础操作独立出来，然后包装一个“流式”处理的库，比如叫pipeline， 这样就可以把重点关注在操作节点以及每个节点的参数上；

库名：cv2lib.py

这里有两个例子：

例子一： 提取人像全身，包括头发；

'''
python person.py
'''

例子二： 提取文本区域；
'''
python getwords.py
'''




