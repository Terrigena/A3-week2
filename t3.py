import cv2
import numpy as np
import os
import requests
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
VISION_API_URL = "https://suite-endpoint-api-apne2.superb-ai.com/endpoints/1df7dfe8-f7b8-4dac-913d-418b886193e4/inference"
TEAM = "kdt2025_1-23"
ACCESS_KEY = "Ci54Olu61E8WMpdbjvbzuaWOmnyY94aw3ayXunSG"
TARGET_SIZE = (700, 700)  # 학습 모델 크기

# 시리얼 통신 설정 (컨베이어 벨트 제어)
ser = serial.Serial("/dev/ttyACM0", 9600)

# 저장 디렉토리 설정
SAVE_DIR = "detected_images"
os.makedirs(SAVE_DIR, exist_ok=True)

EXPECTED_CLASS_COUNTS = {
    "RASPBERRY PICO": 9,
    "USB": 9,
    "OSCILLATOR": 9,
    "CHIPSET": 9,
    "HOLE": 9,
    "BOOTSEL": 9
}

def capture_image():
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Can not open Cam!")
        return None
    ret, img = cam.read()
    cam.release()
    return img

def send_to_api(image):
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
    class_counts = {key: 0 for key in EXPECTED_CLASS_COUNTS.keys()}

    if result and "objects" in result:
        for obj in result["objects"]:
            box = obj["box"]
            label = obj["class"]
            score = obj["score"]

            if label in class_counts:
                class_counts[label] += 1

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

    return image, class_counts

def check_class_counts(class_counts):
    for label, expected_count in EXPECTED_CLASS_COUNTS.items():
        if class_counts.get(label, 0) != expected_count:
            return False
    return True

def save_image(image, filename):
    cv2.imwrite(filename, image)
    print(f"Image saved: {filename}")

def main():
    global ser
    try:
        while True:
            print("Waiting for data from conveyor belt...")
            data = ser.read()
            print(f"Received data: {data}")  # 시리얼 데이터 확인

            if data == b"0":
                print("Conveyor belt stopped. Capturing image...")
                img = capture_image()
                if img is None:
                    print("No image captured.")
                    ser.write(b"1")  # 컨베이어 벨트 다시 시작
                    continue

                print("Image captured. Processing...")
                result, original_width, original_height = send_to_api(img)

                if result:
                    img_with_boxes, class_counts = process_results(img, result, original_width, original_height)
                    is_normal = check_class_counts(class_counts)

                    if is_normal:
                        print("Status: NORMAL")
                        ser.write(b"1")  # 컨베이어 벨트 다시 시작
                    else:
                        print("Status: ABNORMAL")
                        print("Waiting for object removal...")

                        while True:
                            img = capture_image()
                            result, _, _ = send_to_api(img)

                            if result:
                                _, class_counts = process_results(img, result, original_width, original_height)
                                is_normal = check_class_counts(class_counts)

                                if is_normal:
                                    print("Object removed. Resuming conveyor belt.")
                                    ser.write(b"1")  # 컨베이어 벨트 다시 시작
                                    break
                            else:
                                print("Rechecking... No valid response.")

                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    filename = f"{SAVE_DIR}/product_{timestamp}.jpg"
                    save_image(img_with_boxes, filename)
                    cv2.imshow("Detection Results", img_with_boxes)
                else:
                    print("No objects detected.")

                cv2.waitKey(2000)
                cv2.destroyAllWindows()
    except KeyboardInterrupt:
        print("Program terminated by keyboard interrupt.")
    finally:
        ser.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
