import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
def calculate_color_histogram(image):
    
    # 定义直方图的参数
    hist_size = [8, 8, 8]  # 每个通道的直方图bin数量
    hist_ranges = [0, 256, 0, 256, 0, 256]  # 每个通道的像素值范围
    
    # 计算直方图
    hist = cv2.calcHist([image], [0, 1, 2], None, hist_size, hist_ranges)
    
    # 将直方图归一化
    cv2.normalize(hist, hist)
    
    return hist

def compare_histograms(hist1, hist2):
    # 使用巴氏距离度量直方图差异
    similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)
    
    return similarity

subjects = ['arm','light','dog','printer','tree']
data = np.load("imageretrieval/data.npy")  # 读取文件

# 设置路径
path = 'imageretrieval/images_test/'
files = os.listdir(path)

thresholds = []
accuracys = []
thresholds = np.arange(0,1,0.01) #测试阈值
for threshold in thresholds:
    count1 = 0
    counts1 = 0
    count2 = 0
    counts2 = 0
    for file in files:
        # 读取
        image_name = path+file
        test_image = cv2.imread(image_name)

        # 方法：直方图匹配
        test_histogram = calculate_color_histogram(test_image)
        # 在图像数据库中进行图像检索
        results = []
        for i in range(len(data)):
            similarity = compare_histograms(test_histogram, data[i])
            results.append((i, similarity))
        # 根据相似度排序检索结果
        results.sort(key=lambda x: x[1])

        # 输出检索结果
        # for result in results[:10]:
        #     image_index = result[0]
        #     similarity = result[1]
        #     print(subjects[int(image_index/30)], image_index%30+1, "- Similarity:", similarity)

        #相似度最高的类别
        id = int(results[0][0]/30)
        subject = subjects[id]
        similar = results[0][1]
        # print(subject)
        # print(similar)


        #分类正确率计算
        label = file.split('_')[0]
        if(label in subjects): #库中存在的类别
            if(similar<threshold):
                if label == subject:
                    count1 = count1+1
            counts1 = counts1+1
        else: #库中不存在的类别
            if(similar<threshold):
                count2 = count2+1
            counts2 = counts2+1

    print(threshold)
    if(counts1>0):
        accuracy1 = count1/counts1
        print('库中存在的类别的图片被正确识别的概率',accuracy1)
    if(counts2>0):
        accuracy2 = 1 - count2/counts2
        print('库中不存在的类别的图片被正确识别的概率',accuracy2)
    if(counts1>0 and counts2>0):
        accuracy = (count1+counts2-count2)/(counts1+counts2)
        print('所有图片被正确识别的概率',accuracy,'\n')
        
        accuracys.append(accuracy)

accuracy_max = np.max(accuracys)
threshold_max = thresholds[np.argmax(accuracys)]
print('最大准确率：',accuracy_max,'阈值：',threshold_max)
plt.plot(thresholds,accuracys)
plt.xlabel('threshold')
plt.ylabel('accuracy')
plt.show()