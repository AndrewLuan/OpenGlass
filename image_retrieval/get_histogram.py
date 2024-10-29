import cv2
import numpy as np
def calculate_color_histogram(image):
    
    # 定义直方图的参数
    hist_size = [8, 8, 8]  # 每个通道的直方图bin数量
    hist_ranges = [0, 256, 0, 256, 0, 256]  # 每个通道的像素值范围
    
    # 计算直方图
    hist = cv2.calcHist([image], [0, 1, 2], None, hist_size, hist_ranges)
    
    # 将直方图归一化
    cv2.normalize(hist, hist)
    
    return hist


# 读取图像数据库中的图像
database_images = []

path_data = 'imageretrieval/images/'
database_histograms = []
# 计算图像数据库中每个图像的颜色直方图
subjects = ['arm','light','dog','printer','tree']
for i in range(len(subjects)):
    for j in range(30):
        path_tmp = path_data + subjects[i] + str(j+1) + '.jpg'
        database_images.append(cv2.imread(path_tmp))

for image in database_images:
    histogram = calculate_color_histogram(image)
    database_histograms.append(histogram)
np.save("imageretrieval/data.npy",database_histograms)