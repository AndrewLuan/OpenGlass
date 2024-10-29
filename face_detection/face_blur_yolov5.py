import cv2
import os
import torch


model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

def detect_and_mosaic(image, mosaic_size=15,):
    # 使用 YOLOv5 进行人脸检测
    results = model(image)
    for result in results.xyxy[0]: # 遍历检测结果
        x1, y1, x2, y2 = map(int, result[:4])

        # 提取检测到的区域
        region = image[y1:y2, x1:x2]

        # 缩小图像然后放大以产生马赛克效果
        region = cv2.resize(region, (mosaic_size, mosaic_size), interpolation=cv2.INTER_LINEAR)
        region = cv2.resize(region, (x2-x1, y2-y1), interpolation=cv2.INTER_NEAREST)

        # 将马赛克区域放回原图
        image[y1:y2, x1:x2] = region

    return image


def process_image(image_path, output_path = "", mosaic_size=15):
    """
    处理图像文件中的人脸模糊。

    参数:
    image_path (str): 输入图像的路径。
    cascade_path (str): Haar 级联分类器的路径，用于检测人脸或车牌。
    output_path (str): 输出图像的路径。
    mosaic_size (int): 马赛克块的大小，默认为15。
    """
    # 加载图像
    input_dir = os.path.dirname(image_path)
    input_filename = os.path.basename(image_path)
    output_filename = f"blured_{input_filename}"
    output_path = os.path.join(input_dir, output_filename)

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"无法加载图像: {image_path}")


    # 处理图像并保存
    result_image = detect_and_mosaic(image, mosaic_size)
    cv2.imwrite(output_path, result_image)
    print(f"处理完成，保存至: {output_path}")


def process_video(video_path, output_path = "", mosaic_size=15):
    """
    处理视频文件中的人脸模糊。

    参数:
    video_path (str): 输入视频的路径。
    cascade_path (str): Haar 级联分类器的路径，用于检测人脸或车牌。
    output_path (str): 输出视频的路径。
    mosaic_size (int): 马赛克块的大小，默认为15。
    """
    input_dir = os.path.dirname(video_path)
    input_filename = os.path.basename(video_path)
    output_filename = f"blured_{input_filename}"
    output_path = os.path.join(input_dir, output_filename)

    # 加载 Haar 级联分类器

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频文件: {video_path}")

    # 获取视频信息
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fourcc = cv2.VideoWriter_fourcc(*'H264')

    # 初始化视频写入器
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 对当前帧进行人脸检测和马赛克处理
        result_frame = detect_and_mosaic(frame, mosaic_size)

        # 将处理后的帧写入输出视频
        out.write(result_frame)

    # 释放资源
    cap.release()
    out.release()
    print(f"视频处理完成，保存至: {output_path}")


if __name__ == '__main__':
    # 示例用法（图像）
    # image_path = 'demo_face.png'  # 输入图像路径# 人脸检测器路径
    # output_path = f'{image_path}'  # 输出图像路径
    # process_image(image_path, output_path,mosaic_size=15)

    # 示例用法（视频）
    video_path = 'demo_face.mp4'
    process_video(video_path)
