import cv2
import numpy as np

def ArrayToMp4(video_scene, pedal_scene, filename, fps=30):
    """
    NumPy array 리스트를 MP4 비디오로 저장하는 함수

    Args:
        arrays: NumPy array 리스트 (0: 검정, 1: 흰색, shape=(480, 640))
        filename: 저장할 파일 이름 (기본값: "output.mp4")
        fps: 프레임 속도 (기본값: 30)
    """

    height, width = video_scene[0].shape  # 첫 번째 배열의 크기로 설정

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 코덱 설정 (MP4)
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height), isColor=False)

    for i in range(len(video_scene)):
        # 0-1 값을 0-255로 변환하여 uint8 형태로 변환
        frame = (video_scene[i] * 255).astype(np.uint8)
        cv2.putText(frame, "pedal = {}".format(pedal_scene[i]), (10, 50), cv2.FONT_HERSHEY_DUPLEX , 1, (100,0,100))
        out.write(frame)

    out.release()
    print(f"Video saved as {filename}")