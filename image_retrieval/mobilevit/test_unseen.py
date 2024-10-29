import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from load import load_mobilevit_weights
from torch import nn
from torchvision import transforms
transf = transforms.ToTensor()
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

model = load_mobilevit_weights("image_retrieval/mobilevit/MobileViT_S_model_best.pth.tar")
model.eval()
model = nn.Sequential(*list(model.children())[:-1])
subjects_all = ['arm','light','dog','printer','tree','picture','extinguisher','sofa','woodenchair','chair','table']
# data = np.load("imageretrieval/data_test.npy")  # 读取文件

# 设置路径
path = 'image_retrieval/images_test/'
files = os.listdir(path)

thresholds = []
accuracys = []
f1s = []
thresholds = np.arange(0,1,0.01) #测试阈值
TPRS = []
FPRS = []
DATA = []
SUBJECT = []
for i in range(len(subjects_all)):
    subjects = []
    for j in range(len(subjects_all)):
        if(j!=i):
            subjects.append(subjects_all[j])
            
    # 读取图像数据库中的图像
    database_images = []
    path_data = 'image_retrieval/images/'
    database_feat = []
    for i in range(len(subjects)):
        for j in range(30):
            path_tmp = path_data + subjects[i] + str(j+1) + '.jpg'
            database_images.append(cv2.imread(path_tmp))

    for image in database_images:
        image = cv2.resize(image,(256,256))
        image = transf(image)
        image = image.unsqueeze(0)
        feat = model(image).squeeze()
        feat = feat.detach().numpy()
        database_feat.append(feat)
    DATA.append(database_feat)
    SUBJECT.append(subjects)

for threshold in thresholds:
    precision_sum = 0
    recall_sum = 0
    TPR_sum = 0
    FPR_sum = 0
    accuracy_sum = 0
    f1_sum = 0
    for i in range(len(subjects_all)):
        subjects = SUBJECT[i]
        database_feat = DATA[i]
    
        TP = 0
        FP = 0
        TN = 0
        FN = 0

        for file in files:
            # 读取
            image_name = path+file
            test_image = cv2.imread(image_name)
            test_image = cv2.resize(test_image,(256,256))
            test_image = transf(test_image)
            test_image = test_image.unsqueeze(0)
            # 方法：mobilevit
            test_feat = model(test_image).squeeze()
            test_feat = test_feat.detach().numpy()
            # 在图像数据库中进行图像检索
            results = []
            for i in range(len(database_feat)):
                similarity = cosine_similarity(test_feat, database_feat[i])
                results.append((i, similarity))
            # 根据相似度排序检索结果
            results.sort(reverse=True,key=lambda x: x[1])

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
                if(similar>threshold):
                    TP = TP+1
                else:
                    FN = FN+1
        
            else: #库中不存在的类别
                if(similar>threshold):
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

f1_max = np.max(f1s)
threshold_max = thresholds[np.argmax(f1s)]
print('最大f1：',f1_max,'阈值：',threshold_max)

AUC = auc(FPRS, TPRS)
print('AUC',AUC)

plt.figure()
plt.plot(thresholds,f1s)
plt.xlabel('threshold')
plt.ylabel('f1')

plt.figure()
plt.plot(FPRS,TPRS)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.show()