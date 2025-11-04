Edge AI (엣지 AI) 국립순천대학교 컴퓨터교육과 OSS 동아리  
리드미 정리

# 동아리 소개
  Edge Computing과 인공지능 기술을 융합하여 미래 사회에서 활용될 수 있는 지능형 응용 기술을 연구하고 구현하는 것을 목표로 함.  
  
  클라우드 환경에 의존하지 않고 NVIDIA Jetson Nano를 활용해  
  실시간 객체 인식, 추적, 자율주행 등 고속 AI 연산 기술을 탐구함  
  또한 LiDAR 센서, TensorFlow, YOLO, Keras, OpenCV 등 오픈소스를 기반으로  
  AI 이미지 처리 및 센서 융합 기술을 개발하며,  
  이를 통해 실생활에 유용한 AI 응용 시스템을 구현하는 것을 목표로 함.  


# Edge AI 연구 주제 및 정리본


## 챕터 1. 엣지 컴퓨팅 (Edge Computing)

  ### 1-1. 로컬 연산 환경 구축 (On-Device AI Processing)
  Jetson Nano에서 YOLO를 직접 구동해, 클라우드 없이 실시간 추론을 수행함

    ```python
    - Jetson Nano YOLO 실시간 객체 인식
    from ultralytics import YOLO
    import cv2
    
    model = YOLO("yolov5s.pt")
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        results = model(frame)
        cv2.imshow("Edge Detection", results[0].plot())
        if cv2.waitKey(1) == 27: break
    cap.release(); cv2.destroyAllWindows()

  ### 1-2. GPU 병렬 연산 (CUDA Parallelism)

    CUDA 코어를 활용해 수천 개의 연산을 병렬 처리하여 영상 연산 속도를 개선
    
    # CUDA 병렬 연산 예시
    from numba import cuda
    import numpy as np
    
    @cuda.jit
    def add_kernel(a,b,c):
        i = cuda.grid(1)
        if i < a.size:
            c[i] = a[i] + b[i]
    
    a=np.arange(1000,dtype=np.float32)
    b=np.arange(1000,dtype=np.float32)
    c=np.zeros_like(a)
    add_kernel[32,32](a,b,c)
    print(" GPU 패럴 완료 ")

  ### 1-3. 실시간 최적화 (TensorRT & Threading)

    TensorRT를 이용한 FP16 경량화와 스레드 병렬화를 적용함
    
    # TensorRT 모델 최적화
    import torch
    from torch2trt import torch2trt
    from models.common import DetectMultiBackend
    
    model = DetectMultiBackend('yolov5s.pt').eval().cuda()
    dummy = torch.ones((1,3,640,640)).cuda()
    trt_model = torch2trt(model.model, [dummy], fp16_mode=True)


## 챕터 2. 객체 인식 (Object Detection)

  ### 2-1. YOLO 모델 구조 이해

    YOLOv5의 Backbone-Neck-Head 구조와 anchor 기반 탐지 원리를 학습함

    # YOLO 모델 구조 출력
    from ultralytics import YOLO
    model = YOLO("yolov5s.pt")
    model.info()

  ### 2-2. 실시간 객체 인식 구현

    OpenCV DNN을 이용해 실시간 카메라 영상 탐지 수행
    
    import cv2
    net=cv2.dnn.readNet("yolov3.weights","yolov3.cfg")
    cap=cv2.VideoCapture(0)
    while True:
        _,img=cap.read()
        blob=cv2.dnn.blobFromImage(img,1/255,(416,416),swapRB=True)
        net.setInput(blob)
        outs=net.forward(net.getUnconnectedOutLayersNames())

  ### 2-3. 커스텀 학습 및 튜닝

    직접 수집한 데이터셋으로 YOLO 재학습함
    
    python train.py --data custom.yaml --weights yolov5s.pt --epochs 100

## 챕터 3. 객체 추적 (Object Tracking)

  ### 3-1. KCF 알고리즘 이해

    Correlation Filter 기반의 ROI 추적 원리를 학습했습니다.

    import cv2
    tracker=cv2.TrackerKCF_create()
  
  ### 3-2. YOLO + KCF 통합 구현

    YOLO 탐지 결과를 KCF로 연속 추적.

  ### 3-3. 트래커 성능 비교

    CSRT, MOSSE, MedianFlow 등 트래커별 성능 비교.

## 챕터 4. 센서 융합 (Sensor Fusion)

  ### 4-1. LiDAR 데이터 수집

    직렬 통신으로 거리 데이터를 실시간 읽기.
    import serial
    lidar=serial.Serial('/dev/ttyUSB0',115200)
    for _ in range(10):
      print(lidar.readline().decode().strip())

  ### 4-2. 카메라 + LiDAR 융합

    프레임 좌표에 거리 데이터를 매핑해 깊이 맵 구성.
    depth_map[y,x]=lidar_distance[x]

  ###4-3. 환경 인식 테스트

    센서 노이즈 제거 및 인식 정확도 평가.

### 챕터 5. 자율주행 시스템 (Autonomous Driving)

  ### 5-1. 모터 제어

    PWM을 이용한 속도 제어.

    import Jetson.GPIO as GPIO,time
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(17,GPIO.OUT)
    p=GPIO.PWM(17,100); p.start(50)
    time.sleep(1); p.stop(); GPIO.cleanup()

  ### 5-2. 주행 로직 설계

    객체 중심 좌표 기반 방향 제어.
      if target_x < 200: turn_left()
      elif target_x > 440: turn_right()
      else: go_straight()

  ### 5-3. 주행 실험

     자율주행 안정화 및 회피 성공률 90%

