import cv2
import numpy as np
import os
from io import BytesIO
import requests
from PIL import Image
from requests.auth import HTTPBasicAuth
import serial
import time

# 클래스별 색상 정의
CLASS_COLORS = {
    "RASPBERRY PICO": (0, 255, 0),  # 초록색
    "USB": (255, 0, 0),             # 빨간색
    "OSCILLATOR": (0, 0, 255),      # 파란색
    "CHIPSET": (255, 255, 0),       # 노란색
    "HOLE": (255, 0, 255),          # 분홍색
    "BOOTSEL": (0, 255, 255)        # 청록색
}

# API 설정
VISION_API_URL = "https://suite-endpoint-api-apne2.superb-ai.com/endpoints/395f7857-a214-4b1c-874d-e53fa8a96df1/inference"
TEAM = "kdt2025_1-23"
ACCESS_KEY = "Ci54Olu61E8WMpdbjvbzuaWOmnyY94aw3ayXunSG"
TARGET_SIZE = (700, 700)  # 학습 모델 크기
EXPECTED_CLASSES = set(CLASS_COLORS.keys())  # 기대하는 클래스 목록

# 시리얼 통신 설정 (컨베이어 벨트 제어)
ser = serial.Serial("/dev/ttyACM0", 9600)

# 저장 디렉토리 설정
SAVE_DIR = "detected_images"
os.makedirs(SAVE_DIR, exist_ok=True)

def capture_image():
    """카메라로 이미지 캡처."""
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Can not open Cam!")
        return None
    ret, img = cam.read()
    cam.release()
    return img

def send_to_api(image):
    """이미지를 API로 전송하여 감지 결과를 반환."""
    original_height, original_width = image.shape[:2]
    resized_image = cv2.resize(image, TARGET_SIZE)
    _, img_encoded = cv2.imencode(".jpg", resized_image)

    response = requests.post(
        url=VISION_API_URL,
        auth=HTTPBasicAuth(TEAM, ACCESS_KEY),
        headers={"Content-Type": "image/jpeg"},
        data=img_encoded.tobytes(),
    )

    if response.status_code == 200:
        result = response.json()
        return result, original_width, original_height
    else:
        print(f"API Call Fail: {response.status_code}")
        return None, original_width, original_height

def process_results(image, result, original_width, original_height):
    """API 결과를 처리하고 이미지에 객체를 표시."""
    if result and "objects" in result:
        for obj in result["objects"]:
            box = obj["box"]
            label = obj["class"]
            score = obj["score"]

            scale_x = original_width / TARGET_SIZE[0]
            scale_y = original_height / TARGET_SIZE[1]
            x_min = int(box[0] * scale_x)
            y_min = int(box[1] * scale_y)
            x_max = int(box[2] * scale_x)
            y_max = int(box[3] * scale_y)

            color = CLASS_COLORS.get(label, (255, 255, 255))
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
            text = f"{label} ({score:.2f})"
            cv2.putText(image, text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    else:
        print("API NO RESPONSE.")

    return image

def save_image(image, filename):
    """이미지를 저장."""
    cv2.imwrite(filename, image)
    print(f"Image saved: {filename}")

def main():
    global ser  # 전역 변수 ser 사용
    try:
        while True:
            data = ser.read()
            if data == b"0":
                print("Conveyor belt stopped. Capturing image...")
                img = capture_image()
                if img is None:
                    print("No image captured.")
                    ser.write(b"1")  # 컨베이어 벨트 재가동
                    continue

                print("Image captured. Processing...")
                result, original_width, original_height = send_to_api(img)
                img_with_boxes = process_results(img, result, original_width, original_height)

                # 저장 디렉토리 및 파일 이름 설정
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                filename = f"{SAVE_DIR}/product_{timestamp}.jpg"

                # 라벨 박스가 추가된 이미지를 저장
                save_image(img_with_boxes, filename)

                cv2.imshow("Detection Results", img_with_boxes)
                cv2.waitKey(2000)  # 2초 동안 결과 표시
                cv2.destroyAllWindows()

                ser.write(b"1")  # 컨베이어 벨트 재가동
                print("Conveyor belt restarted.")
    except KeyboardInterrupt:
        print("Program terminated by keyboard interrupt.")
    finally:
        ser.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
