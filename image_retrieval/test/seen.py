import cv2
import numpy as np
import os
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

# 读取图像数据库中的图像
database_images = []

path_data = 'imageretrieval/images/'
database_histograms = []
# 计算图像数据库中每个图像的颜色直方图
subjects = ['arm','light','dog','printer','tree','picture','extinguisher','sofa','woodenchair','chair','table']

for i in range(len(subjects)):
    for j in range(30):
        path_tmp = path_data + subjects[i] + str(j+1) + '.jpg'
        database_images.append(cv2.imread(path_tmp))

for image in database_images:
    histogram = calculate_color_histogram(image)
    database_histograms.append(histogram)

# 设置路径
path = 'imageretrieval/images_test/'
files = os.listdir(path)

TP = [0]*len(subjects)
FP = [0]*len(subjects)
TN = [0]*len(subjects)
FN = [0]*len(subjects)

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

    if label == subject:
        TP[subjects.index(subject)] = TP[subjects.index(subject)]+1
        for i in range(len(subjects)):
            if i != subjects.index(subject):
                TN[i] = TN[i]+1
    else:
        FP[subjects.index(subject)] = FP[subjects.index(subject)]+1
        FN[subjects.index(label)] = FN[subjects.index(label)]+1
        for i in range(len(subjects)):
            if i != subjects.index(subject) and i != subjects.index(label):
                TN[i] = TN[i]+1
print(TP)
print(FP)
print(TN)
print(FN)

precision_micro = sum(TP)/(sum(TP)+sum(FP))
recall_micro = sum(TP)/(sum(TP)+sum(FN))
f1_micro = 2*precision_micro*recall_micro/(precision_micro+recall_micro)
accuracy_micro = (sum(TP)+sum(TN))/(sum(TP)+sum(FN)+sum(FP)+sum(TN))

precision_macro = 0
recall_macro = 0
accuracy_macro = 0
for i in range(len(subjects)):
    precision_macro = precision_macro+TP[i]/(TP[i]+FP[i])
    recall_macro = recall_macro + TP[i]/(TP[i]+FN[i])
    accuracy_macro = accuracy_macro + (TP[i]+TN[i])/(TP[i]+FN[i]+FP[i]+TN[i])
precision_macro = precision_macro/len(subjects)
recall_macro = recall_macro/len(subjects)
f1_macro = 2*precision_macro*recall_macro/(precision_macro+recall_macro)
accuracy_macro = accuracy_macro/len(subjects)

print('precision',precision_micro)
print('recall_micro',recall_micro)
print('f1_micro',f1_micro)
print('accuracy_micro',accuracy_micro)

print('precision_macro',precision_macro)
print('recall_macro',recall_macro)
print('f1_macro',f1_macro)
print('accuracy_macro',accuracy_macro)