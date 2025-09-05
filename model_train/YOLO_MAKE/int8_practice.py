import cv2
import numpy as np
import openvino.runtime as ov

# 카메라 장치 번호 (일반적으로 0이 기본 웹캠)
CAMERA_SOURCE = 0
CONFIDENCE_THRESHOLD = 0.25

# --- 사용자 설정 ---
# 1. OpenVINO 모델(.xml)의 전체 경로
OPENVINO_MODEL_PATH = "object_detection_/yolo_results/exp_car_yolov8n/weights/best_int8_openvino_model/best.xml"

# 2. 테스트하고 싶은 이미지의 전체 경로
IMAGE_TO_TEST_PATH = "object_detection_/images/test/_00038.png"
# 3. 학습 시 사용했던 클래스 이름 (순서 중요)
CLASS_NAMES = ["animal", "person", "traffic_red", "traffic_yellow", "traffic_green", "right", "left"]

# 4. 모델 입력 크기 (export 시 설정한 imgsz)
INPUT_SIZE = (320, 320)
# --- 사용자 설정 ---

# 1. OpenVINO 런타임 초기화 및 모델 로드
core = ov.Core()
model = core.read_model(OPENVINO_MODEL_PATH)
compiled_model = core.compile_model(model, "CPU") # 라즈베리파이 CPU에서 실행
input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)

# # 2. 이미지 전처리 (Letterboxing)
# original_image = cv2.imread(IMAGE_TO_TEST_PATH)
# h_orig, w_orig, _ = original_image.shape
#
# # 비율 유지 리사이즈 및 패딩
# w_target, h_target = INPUT_SIZE
# scale = min(h_target / h_orig, w_target / w_orig)
# w_resized, h_resized = int(w_orig * scale), int(h_orig * scale)
# image_resized = cv2.resize(original_image, (w_resized, h_resized))
# padded_image = np.full((h_target, w_target, 3), 114, dtype=np.uint8)
# dw, dh = (w_target - w_resized) // 2, (h_target - h_resized) // 2
# padded_image[dh:h_resized + dh, dw:w_resized + dw, :] = image_resized
#
# # HWC to CHW, BGR to RGB, Normalize
# input_tensor = padded_image.transpose(2, 0, 1) # HWC -> CHW
# input_tensor = np.expand_dims(input_tensor, 0)  # Add batch dimension
# input_tensor = input_tensor.astype(np.float32) / 255.0
#
# # 3. 추론 실행
# results = compiled_model([input_tensor])[output_layer]
#
# # 4. 후처리 (NMS 및 결과 파싱)
# boxes = []
# scores = []
# class_ids = []
# outputs = np.transpose(np.squeeze(results)) # [2100, 11]
#
# for out in outputs:
#     xc, yc, w, h = out[:4]
#     class_id = np.argmax(out[4:])
#     score = out[4 + class_id]
#     if score > 0.25: # Confidence Threshold
#         x1 = (xc - w / 2 - dw) / scale
#         y1 = (yc - h / 2 - dh) / scale
#         x2 = (xc + w / 2 - dw) / scale
#         y2 = (yc + h / 2 - dh) / scale
#         boxes.append([x1, y1, x2-x1, y2-y1])
#         scores.append(score)
#         class_ids.append(class_id)
#
# # NMS (Non-Maximum Suppression) 적용
# indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=0.25, nms_threshold=0.45)
#
# # 5. 최종 결과 시각화
# for i in indices:
#     x, y, w, h = map(int, boxes[i])
#     label = f"{CLASS_NAMES[class_ids[i]]} {scores[i]:.2f}"
#     cv2.rectangle(original_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
#     cv2.putText(original_image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
#
# cv2.imshow("Pure OpenVINO Result", original_image)
# print("\n✅ Result window is showing. Press any key to exit.")
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# 2. 카메라 캡처 객체 생성
cap = cv2.VideoCapture(CAMERA_SOURCE)
if not cap.isOpened():
    print(f"❌ 오류: 카메라를 열 수 없습니다. (장치 번호: {CAMERA_SOURCE})")
    exit()

print("\n✅ Running real-time inference with pure OpenVINO... Press 'q' to exit.")

# 3. 실시간 영상 처리 루프
while True:
    # 카메라에서 프레임 읽기
    success, frame = cap.read()
    if not success:
        print("❌ 오류: 카메라에서 프레임을 읽어올 수 없습니다.")
        break

    # --- 이미지 전처리 (Letterboxing) ---
    h_orig, w_orig, _ = frame.shape
    w_target, h_target = INPUT_SIZE
    scale = min(h_target / h_orig, w_target / w_orig)
    w_resized, h_resized = int(w_orig * scale), int(h_orig * scale)
    image_resized = cv2.resize(frame, (w_resized, h_resized))
    padded_image = np.full((h_target, w_target, 3), 114, dtype=np.uint8)
    dw, dh = (w_target - w_resized) // 2, (h_target - h_resized) // 2
    padded_image[dh:h_resized + dh, dw:w_resized + dw, :] = image_resized

    input_tensor = padded_image.transpose(2, 0, 1)
    input_tensor = np.expand_dims(input_tensor, 0)
    input_tensor = input_tensor.astype(np.float32) / 255.0

    # --- 추론 실행 ---
    results = compiled_model([input_tensor])[output_layer]

    # --- 후처리 ---
    boxes, scores, class_ids = [], [], []
    outputs = np.transpose(np.squeeze(results))

    for out in outputs:
        xc, yc, w, h = out[:4]
        class_id = np.argmax(out[4:])
        score = out[4 + class_id]
        if score > CONFIDENCE_THRESHOLD:
            x1 = (xc - w / 2 - dw) / scale
            y1 = (yc - h / 2 - dh) / scale
            x2 = (xc + w / 2 - dw) / scale
            y2 = (yc + h / 2 - dh) / scale
            boxes.append([x1, y1, x2 - x1, y2 - y1])
            scores.append(score)
            class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=CONFIDENCE_THRESHOLD, nms_threshold=0.45)

    # --- 최종 결과 시각화 ---
    for i in indices:
        x, y, w, h = map(int, boxes[i])
        label = f"{CLASS_NAMES[class_ids[i]]} {scores[i]:.2f}"
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Real-time Pure OpenVINO Result", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 4. 자원 해제
cap.release()
cv2.destroyAllWindows()
print("✅ Inference stopped.")