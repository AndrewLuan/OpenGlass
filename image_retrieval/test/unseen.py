import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import auc
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

subjects_all = ['arm','light','dog','printer','tree','picture','extinguisher','sofa','woodenchair','chair','table']
# data = np.load("imageretrieval/data_test.npy")  # 读取文件

# 设置路径
path = 'imageretrieval/images_test/'
files = os.listdir(path)

thresholds = []
accuracys = []
f1s = []
thresholds = np.arange(0,1,0.01) #测试阈值
TPRS = []
FPRS = []

for threshold in thresholds:
    precision_sum = 0
    recall_sum = 0
    TPR_sum = 0
    FPR_sum = 0
    accuracy_sum = 0
    f1_sum = 0
    for i in range(len(subjects_all)):
        subjects = []
        for j in range(len(subjects_all)):
            if(j!=i):
                subjects.append(subjects_all[j])
                
        # 读取图像数据库中的图像
        database_images = []
        path_data = 'imageretrieval/images/'
        database_histograms = []
        for i in range(len(subjects)):
            for j in range(30):
                path_tmp = path_data + subjects[i] + str(j+1) + '.jpg'
                database_images.append(cv2.imread(path_tmp))

        for image in database_images:
            histogram = calculate_color_histogram(image)
            database_histograms.append(histogram)
    
        TP = 0
        FP = 0
        TN = 0
        FN = 0

        for file in files:
            # 读取
            image_name = path+file
            test_image = cv2.imread(image_name)

            # 方法：直方图匹配
            test_histogram = calculate_color_histogram(test_image)
            # 在图像数据库中进行图像检索
            results = []
            for i in range(len(database_histograms)):
                similarity = compare_histograms(test_histogram, database_histograms[i])
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
                    TP = TP+1
                else:
                    FN = FN+1
        
            else: #库中不存在的类别
                if(similar<threshold):
                    FP = FP+1
                else:
                    TN = TN+1

        # print(TP+FN+FP+TN)
        if (TP == 0):
            recall = 0
            precision = 0
        else:
            recall = TP/(TP+FN)
            precision = TP/(TP+FP)
        if(recall*precision == 0):
            f1 = 0
        else:
            f1 = 2*recall*precision/(recall+precision)
        TPR = TP/(TP+FN)
        FPR = FP/(FP+TN)
        accuracy = (TP+TN)/(TP+FN+FP+TN)

        precision_sum = precision_sum+precision
        recall_sum = recall_sum+recall
        TPR_sum = TPR_sum+TPR
        FPR_sum = FPR_sum+FPR
        accuracy_sum = accuracy_sum+accuracy
        f1_sum = f1_sum+f1
    precision_average = precision_sum/len(subjects_all)
    recall_average = recall_sum/len(subjects_all)
    TPR_average =  TPR_sum/len(subjects_all)
    FPR_average = FPR_sum/len(subjects_all)
    accuracy_average = accuracy_sum/len(subjects_all)
    f1_average = f1_sum/len(subjects_all)

    print(threshold)
    print('recall',recall_average)
    print('precision',precision_average)
    print('f1',f1_average)
    print('TPR',TPR_average)
    print('FPR',FPR_average)
    #print(accuracy)
    print('\n')
    accuracys.append(accuracy)
    TPRS.append(TPR_average)
    FPRS.append(FPR_average)
    f1s.append(f1_average)
# plt.plot(thresholds,accuracys)
# plt.xlabel('threshold')
# plt.ylabel('accuracy')
# accuracy_max = np.max(accuracys)
# threshold_max = thresholds[np.argmax(accuracys)]
# print('最大准确率：',accuracy_max,'阈值：',threshold_max)
# plt.plot(thresholds,accuracys)
# plt.xlabel('threshold')
# plt.ylabel('accuracy')

plt.plot(thresholds,f1s)
plt.xlabel('threshold')
plt.ylabel('f1')
f1_max = np.max(f1s)
threshold_max = thresholds[np.argmax(f1s)]
print('最大f1：',f1_max,'阈值：',threshold_max)
plt.plot(thresholds,f1s)
plt.xlabel('threshold')
plt.ylabel('f1')

AUC = auc(FPRS, TPRS)
print('AUC',AUC)
plt.plot(FPRS,TPRS)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.show()