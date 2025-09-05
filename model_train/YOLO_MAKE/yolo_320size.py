# train_yolo.py
# -----------------------------
# Ultralytics YOLO 학습 및 PTQ 적용 스크립트
# -----------------------------

import os
import random
import shutil
from datetime import datetime
from ultralytics import YOLO

# ========= 사용자 설정 =========
DATASET_DIR = r"./object_detection_"
PROJECT_DIR = r"./object_detection_/yolo_results"
EXP_NAME    = "exp_car_yolov8n"

CLASS_NAMES = [
    "animal", "person", "traffic_red","traffic_yellow", "traffic_green", "right", "left"
]

MODEL_NAME = "./yolov8n.pt"
EPOCHS     = 100
IMGSZ      = 320
BATCH      = 16
LR0        = 0.005
PATIENCE   = 10
DEVICE     = "cpu"

DO_PREDICT_SAMPLES = True
DO_EXPORT_OPENVINO = True # PTQ를 적용할 것이므로 True로 유지
DO_EXPORT_ONNX     = False

PREDICT_SOURCE = os.path.join(DATASET_DIR, "images", "val")
PREDICT_CONF   = 0.25

# ========= 유틸 =========
def ensure_yaml(dataset_dir, class_names):
    """data.yaml 자동 생성 (이미 있으면 덮어쓰지 않음)"""
    yaml_path = os.path.join(dataset_dir, "data.yaml")
    if os.path.exists(yaml_path):
        print(f"[INFO] data.yaml 이미 존재: {yaml_path}")
        return yaml_path

    # 절대 경로로 변환하여 yaml 파일에 기록
    abs_dataset_dir = os.path.abspath(dataset_dir)
    content = [
        f"path: {abs_dataset_dir}", # 절대 경로 사용
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "names:"
    ]
    for i, name in enumerate(class_names):
        content.append(f"  {i}: {name}")

    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write("\n".join(content) + "\n")

    print(f"[OK] data.yaml 생성: {yaml_path}")
    return yaml_path


def sanity_check(dataset_dir):
    """간단 무결성 체크"""
    img_train_dir = os.path.join(dataset_dir, "images", "train")
    lbl_train_dir = os.path.join(dataset_dir, "labels", "train")
    if not os.path.isdir(img_train_dir) or not os.path.isdir(lbl_train_dir):
        print(f"[WARN] train 폴더를 찾을 수 없어 sanity check를 건너뜁니다.")
        return

    missing = []
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    for f in os.listdir(img_train_dir):
        ext = os.path.splitext(f)[1].lower()
        if ext not in exts: continue
        stem = os.path.splitext(f)[0]
        if not os.path.exists(os.path.join(lbl_train_dir, stem + ".txt")):
            missing.append(f)

    if missing:
        print(f"[WARN] 라벨 누락 이미지 {len(missing)}개 예시: {missing[:10]}")
    else:
        print("[OK] 라벨 매칭 이상 없음")


def train():
    # 1) data.yaml 보장
    data_yaml = ensure_yaml(DATASET_DIR, CLASS_NAMES)

    # 2) 간단 체크

    sanity_check(DATASET_DIR)

    # 3) 모델 로드
    print(f"[INFO] Loading model: {MODEL_NAME}")
    model = YOLO(MODEL_NAME)

    # 4) 학습
    print("[INFO] Start training...")
    results = model.train(
        data=data_yaml,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        project=PROJECT_DIR,
        name=EXP_NAME,
        lr0=LR0,
        patience=PATIENCE,
        optimizer="auto",
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        fliplr=0.5,
        mosaic=1.0, mixup=0.15,
        degrees=5, translate=0.05, scale=0.1, shear=0.0, perspective=0.0,
    )

    run_dir = results.save_dir
    best_pt = os.path.join(run_dir, "weights", "best.pt")
    print(f"[OK] Training done. best: {best_pt}")

    # 5) 검증(mAP, PR커브)
    print("[INFO] Validate best weights...")
    model = YOLO(best_pt)
    model.val(data=data_yaml, project=PROJECT_DIR, name=f"{EXP_NAME}_val")

    # 6) 예측 샘플 저장
    if DO_PREDICT_SAMPLES and os.path.exists(PREDICT_SOURCE):
        print(f"[INFO] Predict & save samples from: {PREDICT_SOURCE}")
        model.predict(
            source=PREDICT_SOURCE,
            conf=PREDICT_CONF,
            save=True,
            project=PROJECT_DIR,
            name=f"{EXP_NAME}_pred_val"
        )

     # 7) 내보내기 (PTQ 적용)
    if DO_EXPORT_OPENVINO:
        print("[INFO] Export OpenVINO IR with Data-aware INT8 Quantization (PTQ)...")

        # ✅ 'data'와 'imgsz'를 명시하여 우리 데이터셋에 맞게 양자화를 수행합니다.
        model.export(
            format="openvino",    # OpenVINO 형식으로 내보내기
            int8=True,            # INT8 양자화 활성화
            data=data_yaml,       # 교정 데이터로 우리 val 세트를 사용하도록 지정
            imgsz=IMGSZ,          # 훈련 시와 동일한 이미지 크기로 교정
            half=False,           # FP16 대신 INT8을 목표로 하므로 False
            simplify=True         # ONNX 모델을 단순화하여 호환성 및 속도 향상
        )


    print("[DONE] All finished.")


if __name__ == "__main__":
    train()