# 视频处理与人脸模糊检测项目

## 项目概述

该项目包含多个模块，用于处理视频和图像，主要功能包括视频处理、人脸关键点检测和人脸模糊处理。使用了 `MediaPipe` 和 `OpenCV` 库来实现面部检测、模糊化处理和视频帧展示。

## 项目结构

- **`video_new_api.py`**: 使用 `MediaPipe` 进行人脸关键点检测，并在视频或图像帧上绘制人脸关键点和轮廓。
- **`face_blur.py`**: 使用 Haar 级联分类器检测视频或图像中的人脸，并对其进行模糊处理（马赛克）。
- **`video.py`**: 处理和展示视频帧，并结合 原来老版本api `MediaPipe` 显示人脸网格。

## 安装

在运行项目之前，需要安装以下依赖库：

```bash
pip install -r requirements.txt
```

### 依赖库

- `opencv-python`
- `mediapipe`
- `numpy`
- `tqdm`
- `matplotlib`

## 使用说明

### 1. 人脸关键点检测（`video_new_api.py`）

使用 `MediaPipe` 检测视频中的人脸关键点，并在每一帧上绘制关键点和轮廓。路径在代码中硬编码为 `img_path` 或 `video_path`，可手动修改。

#### 运行方式：

```bash
python video_new_api.py
```

### 2. 人脸模糊处理（`face_blur.py`）

通过 Haar 级联分类器检测视频或图像中的人脸，并对其进行模糊处理。输入和输出路径在代码中硬编码，处理的图像或视频可通过修改路径进行更改。

#### 运行方式：

```bash
python face_blur.py
```

### 3. 视频处理（`video.py`）

结合 `MediaPipe` 处理视频帧并显示人脸网格。输入路径为硬编码的 `video_path`，可以根据需求修改。

#### 运行方式：

```bash
python video.py
```

### 4. 命令行操作（`face.py`）

增加命令行功能 保存路径为输入文件路径下 

#### 运行方式：

```bash
python face.py -t <type of function[blur or detect]> -f <file[image or video]> -i <input_path> 
```

## 配置说明

文件路径和参数在脚本中硬编码。如需处理其他图像或视频文件，需直接修改代码中的路径变量。例如：

```python
image_path = 'your_image.png'
video_path = 'your_video.mp4'
```


