import time
import serial
import os
import cv2

ser = serial.Serial("/dev/ttyACM0", 9600)

def get_img():
    """Get Image From USB Camera

    Returns:
        numpy.array: Image numpy array
    """
    cam = cv2.VideoCapture(0)

    if not cam.isOpened():
        print("Camera Error")
        exit(-1)

    ret, img = cam.read()
    cam.release()

    return img

def crop_img(img, size_dict):
    x = size_dict["x"]
    y = size_dict["y"]
    w = size_dict["width"]
    h = size_dict["height"]
    img = img[y : y + h, x : x + w]
    return img

def save_img(img, folder_path="week2-A3", counter_file="counter.txt"):
    """Save the captured image to a folder with sequential numbering.

    Args:
        img (numpy.array): Image to save.
        folder_path (str): Directory to save the image.
        counter_file (str): File to store the current image counter.
    """
    # 폴더가 없으면 생성
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # 카운터 파일 경로
    counter_path = os.path.join(folder_path, counter_file)

    # 현재 카운터 읽기
    if os.path.exists(counter_path):
        with open(counter_path, "r") as f:
            counter = int(f.read())
    else:
        counter = 0  # 파일이 없으면 0으로 시작

    # 파일 이름 생성
    file_path = os.path.join(folder_path, f"image_{counter}.jpg")

    # 이미지 저장
    cv2.imwrite(file_path, img)
    print(f"Image saved to {file_path}")

    # 카운터 증가 후 저장
    with open(counter_path, "w") as f:
        f.write(str(counter + 1))

try:
    while 1:
        data = ser.read()
        print(data)
        if data == b"0":
            img = get_img()
            crop_info = {"x": 200, "y": 100, "width": 300, "height": 300}

            if crop_info is not None:
                img = crop_img(img, crop_info)

            # 저장 기능 호출
            save_img(img, folder_path="week2-A3")

            cv2.imshow("Captured Image", img)
            cv2.waitKey(2000)  # 2초 동안 사진 표시
            cv2.destroyWindow("Captured Image")  # 2초 후 창 닫기

            ser.write(b"1")
        else:
            pass
except KeyboardInterrupt:
    print("Program terminated by keyboard interrupt.")
finally:
    ser.close()
    cv2.destroyAllWindows()
