import argparse
from video_new_api import generate_image,generate_video
from face_blur_landmarks import blur_image,blur_video


if __name__ == "__main__":
    # 设置命令行参数解析器
    parser = argparse.ArgumentParser(description="detect face and blur face")
    parser.add_argument('-t','--type', type=str, required=True, choices=['detect', 'blur' ], help="function")
    parser.add_argument('-f','--file', type=str, required=True, choices=['image', 'video' ], help="file")
    parser.add_argument('-i','--input_path', type=str, help="Path to the input")

    # 解析命令行参数
    args = parser.parse_args()

    # 根据选择调用相应的函数
    if args.type == 'detect':
        if args.file =='image':
            generate_image(args.input_path)
        elif args.file =='video':
            generate_video(args.input_path)

    elif args.type == 'blur':
        if args.file =='image':
            blur_image(args.input_path)
        elif args.file =='video':
            blur_video(args.input_path)