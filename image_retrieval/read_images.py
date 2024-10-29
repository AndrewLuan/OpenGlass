import cv2
import numpy as np

# 读取图像数据库中的图像
database_images = []

path_data = 'imageretrieval/images/'

subjects = ['arm','light','dog','printer','tree']
for i in range(len(subjects)):
    for j in range(30):
        path_tmp = path_data + subjects[i] + str(j+1) + '.jpg'
        database_images.append(cv2.resize(cv2.imread(path_tmp),(10,10)))

np.save("imageretrieval/data_ssim.npy",database_images)