import cv2
import numpy as np
import os
from load import load_mobilevit_weights
from torch import nn
from torchvision import transforms

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
# 读取图像数据库中的图像
database_images = []
model = load_mobilevit_weights("image_retrieval/mobilevit/MobileViT_S_model_best.pth.tar")
model.eval()
model = nn.Sequential(*list(model.children())[:-1])
path_data = 'image_retrieval/images/'
database_feat = []
# 计算图像数据库中每个图像的feature
subjects = ['arm','light','dog','printer','tree','picture','extinguisher','sofa','woodenchair','chair','table']
transf = transforms.ToTensor()
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
#np.save("image_retrieval/data_feat.npy",database_feat)
# 设置路径
path = 'image_retrieval/images_test/'
files = os.listdir(path)

TP = [0]*len(subjects)
FP = [0]*len(subjects)
TN = [0]*len(subjects)
FN = [0]*len(subjects)
j=0
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
    # if j<3:
    #     for result in results:
    #         image_index = result[0]
    #         similarity = result[1]
    #         print(subjects[int(image_index/30)], image_index%30+1, "- Similarity:", similarity)
    # j = j+1
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