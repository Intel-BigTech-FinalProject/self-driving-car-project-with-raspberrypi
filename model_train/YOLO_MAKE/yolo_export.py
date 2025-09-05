from ultralytics import YOLO

# --- 설정 ---
# 학습 결과로 생성된 best.pt 모델의 정확한 경로
MODEL_PATH = "./object_detection_/yolo_results/exp_car_yolov8n/weights/best.pt"
# 학습 시 사용했던 data.yaml 경로
DATA_YAML_PATH = "./object_detection_/data.yaml"
# 훈련 시 사용했던 이미지 크기
IMGSZ = 320
# --- 설정 ---

print(f"Loading model: {MODEL_PATH}")
model = YOLO(MODEL_PATH)

print("Exporting to OpenVINO with INT8 Quantization (PTQ)...")
model.export(
    format="openvino",
    int8=True,
    data=DATA_YAML_PATH,
    imgsz=IMGSZ,
    workers=0
)

print("\n✅ OpenVINO INT8 conversion complete!")