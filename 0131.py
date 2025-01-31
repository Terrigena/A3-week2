import cv2
import numpy as np
import os
import requests
from requests.auth import HTTPBasicAuth
import serial
import time
import RPi.GPIO as GPIO

# ===================== Servo Motor Configuration =====================
servo_pin = 17  # Servo motor pin number
GPIO.setmode(GPIO.BCM)
GPIO.setup(servo_pin, GPIO.OUT)

pwm = GPIO.PWM(servo_pin, 50)  # 50Hz
pwm.start(0)

def rotate_motor(direction):
    """Rotate the motor forward or backward"""
    try:
        if direction == "forward":
            print("Motor rotating forward...")
            pwm.ChangeDutyCycle(7.5)  # Forward rotation
        elif direction == "backward":
            print("Motor rotating backward...")
            pwm.ChangeDutyCycle(5.5)  # Backward rotation
        time.sleep(2)  # Maintain rotation for 2 seconds
    finally:
        pwm.ChangeDutyCycle(0)  # Stop rotation

# ===================== Image Analysis Configuration =====================
CLASS_COLORS = {
    "USB": (255, 0, 0),
    "OSCILLATOR": (0, 0, 255),
    "CHIPSET": (255, 255, 0),
    "BOOTSEL": (0, 255, 255)
}

VISION_API_URL = "https://suite-endpoint-api-apne2.superb-ai.com/endpoints/79fd669d-8ebc-45ff-85fb-6cdfe8071ba4/inference"
TEAM = "kdt2025_1-23"
ACCESS_KEY = "Ci54Olu61E8WMpdbjvbzuaWOmnyY94aw3ayXunSG"
TARGET_SIZE = (700, 700)  # Resize to 700x700 for API

EXPECTED_CLASS_COUNTS = {
    "USB": 1,
    "OSCILLATOR": 1,
    "CHIPSET": 1,
    "BOOTSEL": 1
}

SAVE_DIR = "detected_images"
os.makedirs(SAVE_DIR, exist_ok=True)

ser = serial.Serial("/dev/ttyACM0", 9600)

# ===================== Main Functions =====================
def capture_image():
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Unable to open camera.")
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
        print(f"API call failed: {response.status_code}")
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
        print("No response from API.")

    return image, class_counts

def check_class_counts(class_counts):
    duplicate_found = any(count > 1 for count in class_counts.values())
    if duplicate_found:
        print("Status: DUPLICATE DETECTED")
        rotate_motor("forward")  # 정방향 회전 후 컨베이어 벨트 가동
        return True
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
            data = ser.read()
            if data == b"0":
                print("Conveyor belt stopped, capturing image...")
                img = capture_image()
                if img is None:
                    print("Failed to capture image.")
                    ser.write(b"1")  # Restart conveyor belt
                    continue

                print("Image captured. Processing...")

                result, original_width, original_height = send_to_api(img)

                if result:
                    img_with_boxes, class_counts = process_results(img, result, original_width, original_height)
                    labeled_filename = f"{SAVE_DIR}/labeled_{time.strftime('%Y%m%d-%H%M%S')}.jpg"
                    save_image(img_with_boxes, labeled_filename)

                    # 3초 동안 라벨링된 이미지 띄우기
                    cv2.imshow("Labeled Image", img_with_boxes)
                    cv2.waitKey(3000)  # 3초 대기
                    cv2.destroyAllWindows()

                    if not check_class_counts(class_counts):
                        print("Status: ABNORMAL")
                        rotate_motor("backward")  # 비정상일 경우 역방향 회전
                else:
                    print("No objects detected.")

                ser.write(b"1")  # Restart conveyor belt
    except KeyboardInterrupt:
        print("Program terminated by keyboard interrupt.")
    finally:
        ser.close()
        pwm.stop()
        GPIO.cleanup()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
