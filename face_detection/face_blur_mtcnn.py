import cv2
import os
from mtcnn import MTCNN


def detect_and_mosaic(image, mosaic_size=15):
    """
    对图像中的人脸进行检测并打上马赛克。

    参数:
    image (numpy.ndarray): 输入图像。
    mosaic_size (int): 马赛克块的大小，默认为15。

    返回:
    numpy.ndarray: 处理后的图像。
    """
    # 初始化 MTCNN 检测器
    detector = MTCNN()

    # 检测人脸
    results = detector.detect_faces(image)

    # 对每个检测到的区域进行马赛克处理
    for result in results:
        # 获取检测到的人脸区域的坐标和大小
        x, y, w, h = result['box']

        # 提取检测到的区域
        region = image[y:y+h, x:x+w]

        # 缩小图像然后放大以产生马赛克效果
        region = cv2.resize(region, (mosaic_size, mosaic_size), interpolation=cv2.INTER_LINEAR)
        region = cv2.resize(region, (w, h), interpolation=cv2.INTER_NEAREST)

        # 将马赛克区域放回原图
        image[y:y+h, x:x+w] = region

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
    image_path = 'demo_face.png'  # 输入图像路径# 人脸检测器路径
    output_path = f'{image_path}'  # 输出图像路径
    process_image(image_path, output_path,mosaic_size=15)

    # 示例用法（视频）
    video_path = 'demo_face.mp4'  # 输入视频路径
    output_video_path = 'blur_demo_face2.mp4'  # 输出视频路径
    process_video(video_path, output_video_path, mosaic_size=15)
