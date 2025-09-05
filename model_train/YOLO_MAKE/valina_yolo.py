import cv2
from ultralytics import YOLO

# from detect_webcam import CAMERA_INDEX

# --- 사용자 설정 ---
# 1. 양자화 전 원본 모델(.pt)의 전체 경로로 수정
PT_MODEL_PATH = "object_detection_/yolo_results/exp_car_yolov8n/weights/best.pt"

# 2. 테스트하고 싶은 이미지의 전체 경로
IMAGE_TO_TEST_PATH = "object_detection_/images/test/_00007.png"

# 3. 학습 시 사용했던 클래스 이름 (순서 중요)
CLASS_NAMES = [
    "animal", "person", "traffic_red", "traffic_yellow", "traffic_green", "right", "left"
]
# --- 사용자 설정 ---

# 1. PyTorch 모델(.pt) 로드
print(f"Loading PyTorch model from: {PT_MODEL_PATH}")
try:
    # .pt 파일은 task를 명시할 필요가 없습니다.
    model = YOLO(PT_MODEL_PATH)
except Exception as e:
    print(f"❌ 모델 로드 중 오류 발생: {e}")
    exit()

# 2. 이미지 예측 실행
# print(f"Running inference on: {IMAGE_TO_TEST_PATH}")
# try:
#     results = model.predict(IMAGE_TO_TEST_PATH, imgsz=320)
# except FileNotFoundError:
#     print(f"❌ 오류: 테스트 이미지를 찾을 수 없습니다. 경로를 확인하세요: {IMAGE_TO_TEST_PATH}")
#     exit()
#
# # 3. 결과 시각화
# print("Processing results...")
# result = results[0]
# original_image = result.orig_img
#
# boxes = result.boxes.xyxy.cpu().numpy()
# class_ids = result.boxes.cls.cpu().numpy()
# confidences = result.boxes.conf.cpu().numpy()
#
# for box, class_id, conf in zip(boxes, class_ids, confidences):
#     x1, y1, x2, y2 = map(int, box)
#     label = f"{CLASS_NAMES[int(class_id)]} {conf:.2f}"
#
#     cv2.rectangle(original_image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 빨간색으로 변경
#     cv2.putText(original_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
#
# cv2.imshow("PyTorch (.pt) Inference Result", original_image)
# print("\n✅ Result window is showing. Press any key to exit.")
# cv2.waitKey(0)
# cv2.destroyAllWindows()



# ---------------------- 카메라 실시간 탐지 ----------------------
CAMERA_SOURCE = 0

# 2. 카메라 캡처 객체 생성
cap = cv2.VideoCapture(CAMERA_SOURCE)
if not cap.isOpened():
    print(f"❌ 오류: 카메라를 열 수 없습니다. (장치 번호: {CAMERA_SOURCE})")
    exit()

print("\n✅ Running real-time inference... Press 'q' to exit.")

# 3. 실시간 영상 처리 루프
while True:
    # 카메라에서 프레임 읽기
    success, frame = cap.read()
    if not success:
        print("❌ 오류: 카메라에서 프레임을 읽어올 수 없습니다.")
        break

    # 현재 프레임으로 예측 실행
    results = model.predict(frame, imgsz=320, verbose=False)  # 실시간 처리이므로 상세 로그는 끔

    # 첫 번째 결과 사용
    result = results[0]

    # 바운딩 박스 정보 추출
    boxes = result.boxes.xyxy.cpu().numpy()
    class_ids = result.boxes.cls.cpu().numpy()
    confidences = result.boxes.conf.cpu().numpy()

    # 프레임에 결과 그리기
    for box, class_id, conf in zip(boxes, class_ids, confidences):
        x1, y1, x2, y2 = map(int, box)
        label = f"{CLASS_NAMES[int(class_id)]} {conf:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 빨간색 박스
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # 결과 프레임을 화면에 보여주기
    cv2.imshow("Real-time PyTorch (.pt) Inference", frame)

    # 'q' 키를 누르면 루프 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 4. 자원 해제
cap.release()
cv2.destroyAllWindows()
print("✅ Inference stopped.")
