import cv2
import numpy as np

def getMSSISM(i1, i2):
    C1 = 6.5025
    C2 = 58.5225
    # INITS
 
    I1 = np.float32(i1) # cannot calculate on one byte large values
    I2 = np.float32(i2)
 
    I2_2 = I2 * I2 # I2^2
    I1_2 = I1 * I1 # I1^2
    I1_I2 = I1 * I2 # I1 * I2
    # END INITS
 
    # PRELIMINARY COMPUTING
    mu1 = cv2.GaussianBlur(I1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(I2, (11, 11), 1.5)
 
    mu1_2 = mu1 * mu1
    mu2_2 = mu2 * mu2
    mu1_mu2 = mu1 * mu2
 
    sigma1_2 = cv2.GaussianBlur(I1_2, (11, 11), 1.5)
    sigma1_2 -= mu1_2
 
    sigma2_2 = cv2.GaussianBlur(I2_2, (11, 11), 1.5)
    sigma2_2 -= mu2_2
 
    sigma12 = cv2.GaussianBlur(I1_I2, (11, 11), 1.5)
    sigma12 -= mu1_mu2
 
    t1 = 2 * mu1_mu2 + C1
    t2 = 2 * sigma12 + C2
    t3 = t1 * t2                    # t3 = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))
 
    t1 = mu1_2 + mu2_2 + C1
    t2 = sigma1_2 + sigma2_2 + C2
    t1 = t1 * t2                    # t1 =((mu1_2 + mu2_2 + C1).*(sigma1_2 + sigma2_2 + C2))
 
    ssim_map = cv2.divide(t3, t1)    # ssim_map =  t3./t1;
 
    mssim = cv2.mean(ssim_map)       # mssim = average of ssim map
    return mssim

database_images = np.load("imageretrieval/data_ssim.npy")
subjects = ['arm','light','dog','printer','tree']

# 读取查询图像
query_image = cv2.resize(cv2.imread('imageretrieval/images/q1.jpg'),(10,10))

# 在图像数据库中进行图像检索
results = []
for i in range(len(database_images)):
    similarity = getMSSISM(query_image, database_images[i])
    similarity = (similarity[0]+similarity[1]+similarity[2])/3
    results.append((i, similarity))

# 根据相似度排序检索结果
results.sort(reverse=True,key=lambda x: x[1])

# 输出检索结果
# for result in results:
#     image_index = result[0]
#     similarity = result[1]
#     print(subjects[int(image_index/30)], image_index%30+1, "- Similarity:", similarity)

#相似度最高的类别

# if(results[0][1]<0.3):
id = int(results[0][0]/30)
print(subjects[id])