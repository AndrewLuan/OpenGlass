import numpy as np
import cv2
from .retrieval import calculate_color_histogram, compare_histograms


def analyze(config):
    query_histogram = calculate_color_histogram(config['photo'])
    if config['pre_image_hist'] is not None:
        sim_with_pre = compare_histograms(query_histogram, config['pre_image_hist'])
        if sim_with_pre < 0.1:
            # print(pre_subject)
            return config['pre_subject'], query_histogram

    results = []
    for i in range(len(config['data'])):
        similarity = compare_histograms(query_histogram, config['data'][i])
        results.append((i, similarity))
    results.sort(key=lambda x: x[1])

    idx = int(results[0][0] / 30)
    print(config['subjects'][idx])
    return config['subjects'][idx], query_histogram


def analyze(config, is_glass):
    if is_glass:
        threshold = config['glass_threshold']
    else:
        threshold = config['esp32_threshold']

    query_histogram = calculate_color_histogram(config['photo'])
    if config['pre_image_hist'] is not None:
        sim_with_pre = compare_histograms(query_histogram, config['pre_image_hist'])
        # print(sim_with_pre)
        if sim_with_pre < threshold:
            results = []
            for i in range(len(config['data'])):
                similarity = compare_histograms(query_histogram, config['data'][i])
                results.append((i, similarity))
            # 根据相似度排序检索结果
            results.sort(key=lambda x: x[1])
            # 相似度最高的类别
            # print(results[0][1])
            if results[0][1] < 0.35:
                id = int(results[0][0] / 30)
                # print(config['subjects'][id])
                return config['subjects'][id], query_histogram
    return None, query_histogram
