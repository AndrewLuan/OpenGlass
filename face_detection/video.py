import cv2
import mediapipe as mp
import time
from tqdm import tqdm
import numpy as np
import os
from mediapipe.framework.formats import landmark_pb2
# os.environ["GLOG_minloglevel"] ="2"
device = 0
# initialize the mediapipe drawing classs
mp_face_mesh = mp.solutions.face_mesh
model = mp_face_mesh.FaceMesh(static_image_mode=False, refine_landmarks=True,
                              max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


# For static images:

def process_frame(img):
    start_time = time.time()
    # Convert the BGR image to RGB and process it with MediaPipe Face Mesh.
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    # img.flags.writeable = False
    results = model.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # Draw face landmarks of each face.
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # 绘制人脸网格
            # https://github.com/googlesamples/mediapipe/blob/main/examples/face_landmarker/
            # 需要进行NormalizedLandmarkList转换
            face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            face_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(
                    x=landmark.x, y=landmark.y, z=landmark.z)
                for landmark in face_landmarks.landmark
            ])
            # print(face_landmarks_proto)
            mp_drawing.draw_landmarks(
                image=img,
                landmark_list=face_landmarks_proto,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                # landmark_drawing_spec为关键点可视化样式，None为默认样式不显示关键点
                # landmark_drawing_spec=mp_drawing.DrawingSpec(color=(66, 77, 229), thickness=1, circle_radius=1),
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_tesselation_style())
            # 绘制脸轮廓
            mp_drawing.draw_landmarks(image=img,
                                      landmark_list=face_landmarks_proto,
                                      connections=mp_face_mesh.FACEMESH_CONTOURS,
                                      landmark_drawing_spec=None,
                                      connection_drawing_spec=mp_drawing_styles
                                      .get_default_face_mesh_contours_style())
            mp_drawing.draw_landmarks(image=img,
                                      landmark_list=face_landmarks_proto,
                                      connections=mp_face_mesh.FACEMESH_IRISES,
                                      landmark_drawing_spec=None,
                                      connection_drawing_spec=mp_drawing_styles
                                      .get_default_face_mesh_iris_connections_style())
    end_time = time.time()
    # print("time cost: ", end_time - start_time)
    fps = 1 / (end_time - start_time)
    img = cv2.putText(img, "FPS: {:.2f}".format(
        fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
    return img


# webcam
def webcam():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            break
        frame = process_frame(image)
        # Flip the image horizontally for a selfie-view display.
        # cv2.imshow('MediaPipe Face Mesh', cv2.flip(image, 1))
        cv2.imshow('MediaPipe Face Mesh', frame)
        if cv2.waitKey(5) & 0xFF == 27:
            break
    cap.release()
    cap.destroyAllWindows()


def generate_video(img_path,output_path):
    file_head = img_path.split("/")[-1]
    output_path = output_path + "output_"+file_head
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
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
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
            frame = process_frame(image)
            cv2.imshow('MediaPipe Face Mesh', frame)
            if cv2.waitKey(50) & 0xFF == 27:
                break
            out.write(frame)
            pbar.update(1)

    cap.release()
    out.release()
    cv2.destroyAllWindows()


# video
def local_display_video(img_path):
    print(img_path)
    cap = cv2.VideoCapture(img_path)
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            break
        frame = process_frame(image)
        cv2.imshow('MediaPipe Face Mesh', frame)
        if cv2.waitKey(50) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # webcam()
    print("test")
    # local_display_video(r"Mediapipe/demo_face.mp4")
    generate_video("demo_face.mp4","./")
