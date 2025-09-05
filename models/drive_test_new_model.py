import threading
import time
import cv2
import RPi.GPIO as GPIO
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from ultralytics import YOLO

# ===================== 모터 설정 =====================
PWMA = 18
AIN1 = 22
AIN2 = 27
PWMB = 23
BIN1 = 25
BIN2 = 24

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(AIN2, GPIO.OUT)
GPIO.setup(AIN1, GPIO.OUT)
GPIO.setup(PWMA, GPIO.OUT)
GPIO.setup(BIN1, GPIO.OUT)
GPIO.setup(BIN2, GPIO.OUT)
GPIO.setup(PWMB, GPIO.OUT)

L_Motor = GPIO.PWM(PWMA, 100)
R_Motor = GPIO.PWM(PWMB, 100)
L_Motor.start(0)
R_Motor.start(0)


def motor_go(speed):
    L_Motor.ChangeDutyCycle(speed)
    GPIO.output(AIN2, True)
    GPIO.output(AIN1, False)
    R_Motor.ChangeDutyCycle(speed)
    GPIO.output(BIN2, True)
    GPIO.output(BIN1, False)


def motor_stop():
    L_Motor.ChangeDutyCycle(0)
    GPIO.output(AIN2, False)
    GPIO.output(AIN1, False)
    R_Motor.ChangeDutyCycle(0)
    GPIO.output(BIN2, False)
    GPIO.output(BIN1, False)


def motor_right(speed):
    L_Motor.ChangeDutyCycle(speed)
    GPIO.output(AIN2, True)
    GPIO.output(AIN1, False)
    R_Motor.ChangeDutyCycle(0)
    GPIO.output(BIN2, False)
    GPIO.output(BIN1, True)


def motor_left(speed):
    L_Motor.ChangeDutyCycle(0)
    GPIO.output(AIN2, False)
    GPIO.output(AIN1, True)
    R_Motor.ChangeDutyCycle(speed)
    GPIO.output(BIN2, True)
    GPIO.output(BIN1, False)


# ===================== 이미지 전처리 =====================
def img_preprocess(image):
    height, _, _ = image.shape
    image = image[int(height / 2):, :, :]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    image = cv2.resize(image, (200, 66))
    image = cv2.GaussianBlur(image, (5, 5), 0)
    return image


def stopline_img_preprocess(image):
    height, _, _ = image.shape
    image = image[int(height / 2):, :, :]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    image = cv2.resize(image, (200, 66))
    image = cv2.GaussianBlur(image, (5, 5), 0)
    image = img_to_array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image


stopline_flag = False


def reset_stopline_flag():
    global stopline_flag
    stopline_flag = False


# ===================== 카메라 설정 =====================
camera = cv2.VideoCapture(0, cv2.CAP_V4L2)
camera.set(3, 320)
camera.set(4, 240)

carState = "stop"
frame = None
lock = threading.Lock()


def capture_frames():
    global frame
    while True:
        ret, image = camera.read()
        if not ret:
            continue
        image = cv2.flip(image, -1)
        with lock:
            frame = image


# ===================== 프레임 처리 =====================
def process_frames():
    global carState, frame
    # 모델 로드
    model = load_model('./mobileNet_model_s.h5')
    stopline_model = load_model('./stopline_model.h5')
    yolo_model = YOLO('./vanila_320.pt')

    try:
        while True:
            with lock:
                if frame is None:
                    continue
                orig_frame = frame.copy()
            
            # --- YOLO 객체 감지 (매 프레임 실행) ---
            class_names = ['animal', 'person', 'traffic_red', 'traffic_yellow', 'traffic_green', 'right', 'left']
            results = yolo_model.predict(orig_frame, conf=0.5, verbose=False)
            
            should_stop = False
            detected_objects_in_frame = []
            
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    cls_name = class_names[cls_id]
                    
                    if cls_name in ['traffic_red', 'animal', 'person']:
                        should_stop = True
                    
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    label = f"{cls_name}:{conf:.2f}"
                    color = (0, 0, 255) if should_stop else (0, 255, 0)
                    cv2.rectangle(orig_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(orig_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    detected_objects_in_frame.append(f"{cls_name} ({conf:.2f})")

            if detected_objects_in_frame:
                print(f"[YOLO Detected]: {', '.join(detected_objects_in_frame)}")

            # --- 모터 제어 우선순위 ---
            if should_stop:
                print("[Safety Stop] 정지 신호, 사람 또는 동물 감지됨. 정지합니다.")
                motor_stop()
            else:
                # --- 정지선 검출 ---
                stopline_X = stopline_img_preprocess(frame)
                stopline_prediction = stopline_model.predict(stopline_X, verbose=False)
                stopline_detected = np.argmax(stopline_prediction[0])
                global stopline_flag
                if stopline_detected and not stopline_flag:
                    print("Stopline detected, stopping for 3 seconds")
                    motor_stop()
                    time.sleep(3)
                    stopline_flag = True
                    threading.Timer(10, reset_stopline_flag).start()
                    continue

                # --- 차선 주행 ---
                preprocessed = img_preprocess(frame)
                preprocessed_array = img_to_array(preprocessed)
                preprocessed_array = preprocessed_array / 255.0
                X = np.asarray([preprocessed_array])
                prediction = model.predict(X, verbose=False)
                steering_angle = prediction[0][0]
                
                if carState == "go":
                    if 70 <= steering_angle <= 100:
                        speedSet = 43
                        motor_go(speedSet)
                    elif steering_angle > 100:
                        speedSet = 31
                        motor_right(speedSet)
                    elif steering_angle < 70:
                        speedSet = 35
                        motor_left(speedSet)
                else:
                    motor_stop()

            # --- 화면 출력 ---
            cv2.imshow('Original Frame + YOLO', orig_frame)
            # cv2.imshow('Preprocessed Frame', preprocessed) # 필요 시 주석 해제

            keyValue = cv2.waitKey(1)
            if keyValue == ord('q'):
                break
            elif keyValue == 82:
                print("go")
                carState = "go"
            elif keyValue == 84:
                print("stop")
                carState = "stop"

        cv2.destroyAllWindows()
    except KeyboardInterrupt:
        pass


# ===================== 메인 =====================
def main():
    capture_thread = threading.Thread(target=capture_frames)
    process_thread = threading.Thread(target=process_frames)
    capture_thread.start()
    process_thread.start()
    capture_thread.join()
    process_thread.join()


if __name__ == '__main__':
    try:
        main()
    finally:
        GPIO.cleanup()
        camera.release()