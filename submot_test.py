import RPi.GPIO as GPIO
import time

servo_pin = 17  # 서보모터 핀 번호

# GPIO 설정
GPIO.setmode(GPIO.BCM)
GPIO.setup(servo_pin, GPIO.OUT)

# PWM 객체 생성 (50Hz)
pwm = GPIO.PWM(servo_pin, 50)
pwm.start(0)  # 초기 듀티 사이클 설정

def set_angle(angle):
    duty = 2 + (angle / 18)  # 각도에 따른 듀티 사이클 계산
    GPIO.output(servo_pin, True)
    pwm.ChangeDutyCycle(duty)
    time.sleep(0.5)
    GPIO.output(servo_pin, False)
    pwm.ChangeDutyCycle(0)

try:
    while True:
        for angle in range(0, 181, 10):  # 0도에서 180도까지 이동
            set_angle(angle)
        for angle in range(180, -1, -10):  # 180도에서 0도까지 이동
            set_angle(angle)
except KeyboardInterrupt:
    pwm.stop()
    GPIO.cleanup()
