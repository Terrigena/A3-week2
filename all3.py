import cv2
import numpy as np
import os
import requests
from requests.auth import HTTPBasicAuth
import serial
import time
import RPi.GPIO as GPIO

# ===================== 서보모터 설정 =====================
servo_pin = 17  # 서보모터 핀 번호
GPIO.setmode(GPIO.BCM)
GPIO.setup(servo_pin, GPIO.OUT)

pwm = GPIO.PWM(servo_pin, 50)  # 50Hz
pwm.start(0)

def rotate_motor(direction):
    """모터를 정방향 또는 역방향으로 회전"""
    try:
        if direction == "forward":
            print("모터 정방향 회전 중...")
            pwm.ChangeDutyCycle(7.5)  # 정방향 회전
        elif direction == "backward":
            print("모터 역방향 회전 중...")
            pwm.ChangeDutyCycle(5.5)  # 역방향 회전
        time.sleep(2)  # 회전 유지 시간
    finally:
        pwm.ChangeDutyCycle(0)  # 회전 중지

# ===================== 이미지 분석 설정 =====================
CLASS_COLORS = {
    "RASPBERRY PICO": (0, 255, 0),
    "USB": (255, 0, 0),
    "OSCILLATOR": (0, 0, 255),
    "CHIPSET": (255, 255, 0),
    "HOLE": (255, 0, 255),
    "BOOTSEL": (0, 255, 255)
}

VISION_API_URL = "https://suite-endpoint-api-apne2.superb-ai.com/endpoints/79fd669d-8ebc-45ff-85fb-6cdfe8071ba4/inference"
TEAM = "kdt2025_1-23"
ACCESS_KEY = "Ci54Olu61E8WMpdbjvbzuaWOmnyY94aw3ayXunSG"
TARGET_SIZE = (700, 700)

EXPECTED_CLASS_COUNTS = {
    "RASPBERRY PICO": 9,
    "USB": 9,
    "OSCILLATOR": 9,
    "CHIPSET": 9,
    "HOLE": 9,
    "BOOTSEL": 9
}

SAVE_DIR = "detected_images"
os.makedirs(SAVE_DIR, exist_ok=True)

ser = serial.Serial("/dev/ttyACM0", 9600)

# ===================== 주요 함수 =====================
def capture_image():
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("카메라를 열 수 없습니다.")
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
        print(f"API 호출 실패: {response.status_code}")
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
        print("API 응답 없음.")

    return image, class_counts

def check_class_counts(class_counts):
    for label, expected_count in EXPECTED_CLASS_COUNTS.items():
        if class_counts.get(label, 0) != expected_count:
            return False
    return True

def save_image(image, filename):
    cv2.imwrite(filename, image)
    print(f"이미지가 저장되었습니다: {filename}")

def main():
    global ser
    try:
        while True:
            data = ser.read()
            if data == b"0":
                print("컨베이어 벨트 정지, 이미지 캡처 중...")
                img = capture_image()
                if img is None:
                    print("이미지를 캡처하지 못했습니다.")
                    ser.write(b"1")  # 컨베이어 벨트 재가동
                    continue

                print("이미지 캡처 완료. 분석 중...")
                result, original_width, original_height = send_to_api(img)

                if result:
                    img_with_boxes, class_counts = process_results(img, result, original_width, original_height)
                    is_normal = check_class_counts(class_counts)

                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    filename = f"{SAVE_DIR}/product_{timestamp}.jpg"
                    save_image(img_with_boxes, filename)

                    if is_normal:
                        print("상태: 정상")
                        ser.write(b"1")  # 정상일 경우 컨베이어 벨트만 재가동
                    else:
                        print("상태: 비정상")
                        rotate_motor("backward")  # 비정상일 경우 역방향 회전

                    cv2.imshow("Detection Results", img_with_boxes)
                else:
                    print("객체를 감지하지 못했습니다.")

                cv2.waitKey(2000)  # 2초간 결과 표시
                cv2.destroyAllWindows()

                ser.write(b"1")  # 컨베이어 벨트 재가동
    except KeyboardInterrupt:
        print("프로그램이 키보드 인터럽트로 종료되었습니다.")
    finally:
        ser.close()
        pwm.stop()
        GPIO.cleanup()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
