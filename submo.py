import RPi.GPIO as GPIO
import time

servo_pin = 17  # 서보모터 핀 번호

# GPIO 설정
GPIO.setmode(GPIO.BCM)
GPIO.setup(servo_pin, GPIO.OUT)

# PWM 객체 생성 (50Hz)
pwm = GPIO.PWM(servo_pin, 50)
pwm.start(0)  # 초기 듀티 사이클 설정

def rotate(direction):
    """
    direction에 따라 정방향 또는 역방향으로 회전
    정방향: "forward"
    역방향: "backward"
    """
    if direction == "forward":
        print("정방향 회전 중...")
        pwm.ChangeDutyCycle(7.5)  # 정방향 회전 (값은 서보모터에 따라 조정 필요)
    elif direction == "backward":
        print("역방향 회전 중...")
        pwm.ChangeDutyCycle(5.5)  # 역방향 회전 (값은 서보모터에 따라 조정 필요)
    time.sleep(2)  # 2초 동안 회전 (360도 회전 시간 조정)
    pwm.ChangeDutyCycle(0)  # 회전 중지

try:
    while True:
        # 신호 입력 받기 (예: 사용자 입력으로 신호 테스트)
        signal = input("신호를 입력하세요 (forward/backward/exit): ").strip().lower()

        if signal == "forward":
            rotate("forward")
        elif signal == "backward":
            rotate("backward")
        elif signal == "exit":
            print("프로그램 종료")
            break
        else:
            print("잘못된 입력입니다. 'forward', 'backward' 또는 'exit'을 입력하세요.")

except KeyboardInterrupt:
    print("\n프로그램 종료")
finally:
    pwm.stop()
    GPIO.cleanup()
