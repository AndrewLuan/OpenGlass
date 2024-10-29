import cv2
import numpy as np


def calculate_color_histogram(image):
    # 将图像转换为HSV颜色空间
    # hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

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


def load_database():
    # 读取图像数据库中的图像
    database_images = []
    # database_images.append(cv2.imread('imageretrieval/image1.jpg'))
    # database_images.append(cv2.imread('imageretrieval/image2.jpg'))
    # database_images.append(cv2.imread('imageretrieval/image3.jpg'))
    path_data = 'imageretrieval/images1/'
    database_histograms = []
    # 计算图像数据库中每个图像的颜色直方图
    subjects = ['arm', 'light', 'dog']

    for i in range(len(subjects)):
        for j in range(10):
            path_tmp = path_data + subjects[i] + str(j + 1) + '.jpg'
            database_images.append(cv2.imread(path_tmp))

    for image in database_images:
        histogram = calculate_color_histogram(image)
        database_histograms.append(histogram)


if __name__ == "__main__":
    # 读取查询图像
    query_image = cv2.imread('imageretrieval/images/q1.jpg')
    # 计算查询图像的颜色直方图
    query_histogram = calculate_color_histogram(query_image)
    # 在图像数据库中进行图像检索
    results = []
    for i in range(len(database_images)):
        similarity = compare_histograms(query_histogram, database_histograms[i])
        results.append((i, similarity))
    # 根据相似度排序检索结果
    results.sort(key=lambda x: x[1])

    # # 输出检索结果
    # for result in results:
    #     image_index = result[0]
    #     similarity = result[1]
    #     print("Image", subjects[int(image_index/10)], "- Similarity:", similarity)

    # 相似度前十个图片中最多的类别
    counts = [0] * len(subjects)
    for i in range(10):
        result = results[i]
        image_index = result[0]
        similarity = result[1]
        counts[int(image_index / 10)] = counts[int(image_index / 10)] + 1
    id = np.argmax(counts)
    print(subjects[id])
