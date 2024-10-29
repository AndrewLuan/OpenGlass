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

def compare_histograms(hist1, hist2):
    # 使用巴氏距离度量直方图差异
    similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)
    
    return similarity


if __name__ == "__main__":
    subjects = ['arm','light','dog','printer','tree']
    data = np.load("image_retrieval/data.npy")  # 读取文件
    # 读取查询图像
    query_image = cv2.imread('imageretrieval/images/q50.jpg')
    # 计算查询图像的颜色直方图
    query_histogram = calculate_color_histogram(query_image)
    # 在图像数据库中进行图像检索
    results = []
    for i in range(len(data)):
        similarity = compare_histograms(query_histogram, data[i])
        results.append((i, similarity))
    # 根据相似度排序检索结果
    results.sort(key=lambda x: x[1])

    # 输出检索结果
    for result in results:
        image_index = result[0]
        similarity = result[1]
        print("Image", subjects[int(image_index/30)], "- Similarity:", similarity)

    #相似度最高的类别
# #相似度最高的类别

    # if(results[0][1]<0.3):
    # id = int(results[0][0]/30)
    # print(subjects[id])
