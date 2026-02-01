# Endoscopic Images Computer Vision Project - Mask R-CNN

## 프로젝트 개요
AI Hub 내시경 이미지 합성 데이터셋을 활용한 Mask R-CNN 기반 병변 탐지, 분류, 분할 시스템

## 데이터셋 정보
- **출처**: AI Hub - 내시경 이미지 합성데이터
- **규모**: 총 40,000장 (위 20,000장, 대장 20,000장)
- **클래스**: 궤양(Ulcer), 용종(Polyp), 암(Cancer)
- **어노테이션**: Bounding Box + Segmentation Mask
- **이미지 해상도**: 2048x2048 PNG

## 프로젝트 구조
```
maskrcnn_endoscopy_project/
├── config/                     # 설정 파일
│   ├── config.py              # 메인 설정
│   └── model_config.py        # 모델 하이퍼파라미터
├── data/                      # 데이터 관련
│   ├── dataset.py             # Dataset 클래스
│   ├── transforms.py          # 데이터 증강
│   └── loader.py              # DataLoader 설정
├── models/                    # 모델 정의
│   ├── maskrcnn.py           # Mask R-CNN 모델
│   └── backbone.py           # 백본 네트워크
├── utils/                     # 유틸리티
│   ├── logger.py             # 로깅
│   ├── metrics.py            # 평가 메트릭
│   ├── visualize.py          # 시각화
│   └── checkpoint.py         # 체크포인트 관리
├── train.py                   # 학습 스크립트
├── inference.py               # 추론 스크립트
├── evaluate.py                # 평가 스크립트
├── requirements.txt           # 의존성 패키지
└── README.md                  # 프로젝트 설명
```

## 설치 및 환경 설정

### 1. 가상환경 생성 (권장)
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows
```

### 2. 필요 패키지 설치
```bash
pip install -r requirements.txt
```

### 3. 데이터 준비
AI Hub에서 다운로드한 데이터를 다음 구조로 배치:
```
data/
├── train/
│   ├── images/
│   └── annotations/
├── val/
│   ├── images/
│   └── annotations/
└── test/
    ├── images/
    └── annotations/
```

## 사용법

### 학습
```bash
python train.py --config config/config.py --epochs 100 --batch_size 4
```

### 평가
```bash
python evaluate.py --checkpoint checkpoints/best_model.pth --data_dir data/test
```

### 추론
```bash
python inference.py --checkpoint checkpoints/best_model.pth --image_path sample.png
```

## 주요 기능

### 1. 데이터 처리
- JSON 어노테이션 파싱
- Polygon → Mask 변환
- 멀티클래스 라벨링 (장기, 병변 종류)

### 2. 모델
- Mask R-CNN (ResNet-50/101 백본)
- Transfer Learning (COCO 사전학습 가중치)
- Instance Segmentation

### 3. 학습 전략
- Data Augmentation (Rotation, Flip, Color Jitter)
- Learning Rate Scheduler
- Early Stopping
- Mixed Precision Training

### 4. 평가 메트릭
- mAP (mean Average Precision)
- IoU (Intersection over Union)
- F1 Score
- Confusion Matrix

## 성능 목표
- Detection mAP@0.5: > 0.75
- Segmentation mAP@0.5: > 0.70
- 추론 속도: < 200ms/image (GPU)

## 라이선스 및 주의사항
- 본 데이터는 AI Hub의 이용약관을 준수해야 합니다
- 개인정보보호법에 의해 제3자 이전 및 원본데이터 추론 금지
- 연구 및 교육 목적으로만 사용 가능

## 참고자료
- [Mask R-CNN Paper](https://arxiv.org/abs/1703.06870)
- [Detectron2 Documentation](https://detectron2.readthedocs.io/)
- [AI Hub Dataset](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71666)

## 문의
프로젝트 관련 문의사항은 GitHub Issues를 통해 남겨주세요.
