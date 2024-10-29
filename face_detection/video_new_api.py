# @markdown We implemented some functions to visualize the face landmark detection results. <br/> Run the following cell to activate the functions.
import cv2
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from tqdm import tqdm
import time
from pprint import pprint
import os


def draw_landmarks_on_image(rgb_image, detection_result):
    face_landmarks_list = detection_result.face_landmarks
    #转化为BGR
    annotated_image = cv2.cvtColor(np.copy(rgb_image), cv2.COLOR_RGB2BGR)
    # Loop through the detected faces to visualize.
    for idx in range(len(face_landmarks_list)):
        face_landmarks = face_landmarks_list[idx]

        # Draw the face landmarks.
        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
        ])

        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_tesselation_style())
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_contours_style())
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_iris_connections_style())
        
    return annotated_image


def plot_face_blendshapes_bar_graph(face_blendshapes):
    # Extract the face blendshapes category names and scores.
    face_blendshapes_names = [
        face_blendshapes_category.category_name for face_blendshapes_category in face_blendshapes]
    face_blendshapes_scores = [
        face_blendshapes_category.score for face_blendshapes_category in face_blendshapes]
    # The blendshapes are ordered in decreasing score value.
    face_blendshapes_ranks = range(len(face_blendshapes_names))

    fig, ax = plt.subplots(figsize=(12, 12))
    bar = ax.barh(face_blendshapes_ranks, face_blendshapes_scores,
                  label=[str(x) for x in face_blendshapes_ranks])
    ax.set_yticks(face_blendshapes_ranks, face_blendshapes_names)
    ax.invert_yaxis()

    # Label each bar with values
    for score, patch in zip(face_blendshapes_scores, bar.patches):
        plt.text(patch.get_x() + patch.get_width(),
                 patch.get_y(), f"{score:.4f}", va="top")

    ax.set_xlabel('Score')
    ax.set_title("Face Blendshapes")
    plt.tight_layout()
    plt.show()


def webcam():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            break
        frame = draw_landmarks_on_image(image)
        # Flip the image horizontally for a selfie-view display.
        # cv2.imshow('MediaPipe Face Mesh', cv2.flip(image, 1))
        cv2.imshow('MediaPipe Face Mesh', frame)
        if cv2.waitKey(5) & 0xFF == 27:
            break
    cap.release()
    cap.destroyAllWindows()


def generate_image(img_path):

    input_dir = os.path.dirname(img_path)
    input_filename = os.path.basename(img_path)
    output_filename = f"detected_{input_filename}"
    output_path = os.path.join(input_dir, output_filename)

    # Load the input image from an image file.
    mp_image = mp.Image.create_from_file(img_path)

    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path='face_landmarker_v2_with_blendshapes.task'),
        running_mode=VisionRunningMode.IMAGE)

    with FaceLandmarker.create_from_options(options) as landmarker:
        # Perform face landmarking on the provided single image.
        face_landmarker_result = landmarker.detect(mp_image)

        # 打印对象的所有键
        # print("Keys in face_landmarker_result:")
        # print(list(vars(face_landmarker_result).keys()))

        # 绘制并显示带有标记的图像
        annotated_image = draw_landmarks_on_image(
            mp_image.numpy_view(), face_landmarker_result)
        #保存新图片
        cv2.imwrite(output_path, annotated_image)
        # # 将图像显示在一个可调整大小的窗口中

        # window_name = 'Annotated Image'
        # cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        # cv2.imshow(window_name, annotated_image)

        # # 让用户手动调整窗口大小
        # cv2.resizeWindow(window_name, 800, 600)  # 可以设置成你想要的初始大小
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # # 打印面部特征点
        # print(face_landmarker_result.face_landmarks)


def generate_video(img_path):
    base_options = python.BaseOptions(
        model_asset_path='face_landmarker_v2_with_blendshapes.task')
    VisionRunningMode = mp.tasks.vision.RunningMode
    options = vision.FaceLandmarkerOptions(base_options=base_options,
                                           output_face_blendshapes=True,
                                           output_facial_transformation_matrixes=True,
                                           num_faces=1,
                                           running_mode=VisionRunningMode.VIDEO
                                           )
    detector = vision.FaceLandmarker.create_from_options(options)

    input_dir = os.path.dirname(img_path)
    input_filename = os.path.basename(img_path)
    output_filename = f"detected_{input_filename}"
    output_path = os.path.join(input_dir, output_filename)

    cap = cv2.VideoCapture(img_path)
    frame_count = 0
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            frame_count += 1
        else:
            break
    cap.release()
    cap = cv2.VideoCapture(img_path)
    # frame_count=capture.get(7) # 7代表CAP_PROP_FRAME_COUNT
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    # frame_size=(int(cap.get(3)),int(cap.get(4)))
    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                  int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    # fps=cap.get(5)    # 5代表CAP_PROP_FPS
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
    with tqdm(total=frame_count-1) as pbar:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                break
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
            frame_timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            results = detector.detect_for_video(mp_image, frame_timestamp_ms)
            frame = draw_landmarks_on_image(cv2.cvtColor(image,cv2.COLOR_BGR2RGB), results)
            # cv2.imshow("jpg",cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
            # cv2.imshow('MediaPipe Face Mesh', frame)
            if cv2.waitKey(50) & 0xFF == 27:
                break
            out.write(frame)
            pbar.update(1)
    cap.release()
    out.release()
    cv2.destroyAllWindows()


# video
def local_display_video(img_path):
    # print(img_path)
    cap = cv2.VideoCapture(img_path)
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            break
        frame = draw_landmarks_on_image(image)
        cv2.imshow('MediaPipe Face Mesh', frame)
        if cv2.waitKey(50) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    print("test")
    # img_path = "./demo_face.png"
    # generate_image(img_path,)

    video_path = "./demo_face.mp4"
    generate_video(video_path)
    # plot_face_blendshapes_bar_graph(detection_result.face_blendshapes[0])
    # print(detection_result.facial_transformation_matrixes)

