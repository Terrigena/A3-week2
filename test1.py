import cv2
import numpy as np
import requests
from requests.auth import HTTPBasicAuth
import serial
import time
import threading

# 클래스별 색상 정의
CLASS_COLORS = {
    "RASPBERRY PICO": (0, 255, 0),
    "USB": (255, 0, 0),
    "OSCILLATOR": (0, 0, 255),
    "CHIPSET": (255, 255, 0),
    "HOLE": (255, 0, 255),
    "BOOTSEL": (0, 255, 255)
}

# API 설정
VISION_API_URL = "https://suite-endpoint-api-apne2.superb-ai.com/endpoints/79fd669d-8ebc-45ff-85fb-6cdfe8071ba4/inference"
TEAM = "kdt2025_1-23"
ACCESS_KEY = "Ci54Olu61E8WMpdbjvbzuaWOmnyY94aw3ayXunSG"
TARGET_SIZE = (700, 700)  # 학습 모델 크기
EXPECTED_CLASSES = set(CLASS_COLORS.keys())  # 기대하는 클래스 목록

# 시리얼 통신 설정 (컨베이어 벨트 제어)
ser = serial.Serial("/dev/ttyACM0", 9600)

# 전역 변수
latest_frame = None
result_queue = []
lock = threading.Lock()


def capture_video():
    """카메라로 이미지를 캡처하는 스레드."""
    global latest_frame
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Cannot open camera!")
        return

    while True:
        ret, frame = cam.read()
        if ret:
            with lock:
                latest_frame = frame
        time.sleep(0.03)  # 30 FPS

    cam.release()


def send_to_api_thread():
    """API 호출을 처리하는 스레드."""
    global result_queue
    while True:
        with lock:
            if latest_frame is not None:
                image = latest_frame.copy()
            else:
                continue

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
            with lock:
                result_queue.append((result, original_width, original_height))
        else:
            print(f"API Call Fail: {response.status_code}")

        time.sleep(0.5)  # API 호출 간격


def process_results(image, result, original_width, original_height):
    """API 결과를 처리하고 이미지에 객체를 표시."""
    detected_classes = set()

    if result and "objects" in result:
        for obj in result["objects"]:
            box = obj["box"]
            label = obj["class"]
            score = obj["score"]

            detected_classes.add(label)

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

    return image, detected_classes


def main():
    is_moving = True

    # 스레드 시작
    threading.Thread(target=capture_video, daemon=True).start()
    threading.Thread(target=send_to_api_thread, daemon=True).start()

    while True:
        if is_moving and result_queue:
            with lock:
                result, original_width, original_height = result_queue.pop(0)

            img_with_boxes, detected_classes = process_results(latest_frame, result, original_width, original_height)

            cv2.imshow("Detection Results", img_with_boxes)

            if result and "objects" in result and len(result["objects"]) > 0:
                print("Object Detected. Stopping Conveyor Belt.")
                ser.write(b"STOP\n")
                time.sleep(0.5)
                is_moving = False
                if not EXPECTED_CLASSES.issubset(detected_classes):
                    print("Defective Product Detected!")

        elif not is_moving and result_queue:
            with lock:
                result, original_width, original_height = result_queue.pop(0)

            if result and ("objects" not in result or len(result["objects"]) == 0):
                print("Object Removed. Starting Conveyor Belt.")
                ser.write(b"START\n")
                time.sleep(0.5)
                is_moving = True

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
