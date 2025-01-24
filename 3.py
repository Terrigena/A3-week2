import cv2
import gradio as gr
import requests
import numpy as np
from PIL import Image
from requests.auth import HTTPBasicAuth

# 클래스별 색상 정의
CLASS_COLORS = {
    "RASPBERRY PICO": (0, 255, 0),  # 초록색
    "USB": (255, 0, 0),             # 빨간색
    "OSCILLATOR": (0, 0, 255),      # 파란색
    "CHIPSET": (255, 255, 0),       # 노란색
    "HOLE": (255, 0, 255),          # 분홍색
    "BOOTSEL": (0, 255, 255)        # 청록색
}

# 가상의 비전 AI API URL 및 인증 정보
VISION_API_URL = "https://suite-endpoint-api-apne2.superb-ai.com/endpoints/79fd669d-8ebc-45ff-85fb-6cdfe8071ba4/inference"
TEAM = "kdt2025_1-23"
ACCESS_KEY = "Ci54Olu61E8WMpdbjvbzuaWOmnyY94aw3ayXunSG"


def process_image(image):
    # 이미지를 OpenCV 형식으로 변환
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # 이미지를 API에 전송할 수 있는 형식으로 변환
    _, img_encoded = cv2.imencode(".jpg", image)

    # API 호출
    response = requests.post(
        url=VISION_API_URL,
        auth=HTTPBasicAuth(TEAM, ACCESS_KEY),
        headers={"Content-Type": "image/jpeg"},
        data=img_encoded.tobytes(),
    )

    # API 응답 처리
    if response.status_code == 200:
        try:
            result = response.json()
            if "objects" in result:
                # API 결과를 바탕으로 박스 그리기
                for obj in result["objects"]:
                    box = obj["box"]  # bounding box 좌표
                    label = obj["class"]  # 객체의 이름
                    score = obj["score"]  # 신뢰도

                    # 박스 좌표 변환: box = [x_min, y_min, x_max, y_max]
                    x_min, y_min, x_max, y_max = box
                    start_point = (int(x_min), int(y_min))  # 좌상단 좌표
                    end_point = (int(x_max), int(y_max))    # 우하단 좌표

                    # 클래스별 색상 선택 (기본값: 흰색)
                    color = CLASS_COLORS.get(label, (255, 255, 255))

                    # 박스 그리기
                    cv2.rectangle(image, start_point, end_point, color, 2)

                    # 텍스트 추가 (클래스 이름과 신뢰도)
                    text = f"{label} ({score:.2f})"
                    position = (int(x_min), int(y_min) - 10)  # 텍스트 위치
                    cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            else:
                print("API 응답에 'objects' 키가 없습니다.")
        except ValueError:
            print("API 응답을 JSON으로 변환하는 데 실패했습니다.")
    else:
        print(f"API 호출 실패: {response.status_code}, {response.text}")

    # BGR 이미지를 RGB로 변환
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(image)


# Gradio 인터페이스 설정
iface = gr.Interface(
    fn=process_image,
    inputs=gr.Image(type="pil"),
    outputs="image",
    title="Vision AI Object Detection",
    description="Upload an image to detect objects using Vision AI.",
)

# 인터페이스 실행
iface.launch(share=True)
