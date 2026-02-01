# Mask R-CNN 내시경 이미지 프로젝트 사용 가이드

## 1. 환경 설정

### 1.1 가상환경 생성 및 활성화
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows
```

### 1.2 패키지 설치
```bash
# 기본 패키지 설치
pip install -r requirements.txt

# Detectron2 설치 (선택사항 - 고급 기능 사용 시)
pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

## 2. 데이터 준비

### 2.1 AI Hub 데이터 다운로드
1. https://aihub.or.kr 접속
2. "내시경 이미지 합성데이터" 검색
3. 데이터 다운로드 (로그인 필요)

### 2.2 데이터 구조
다운로드한 데이터를 다음 구조로 정리:
```
raw_data/
├── images/
│   ├── 1_1_00001.png
│   ├── 1_1_00002.png
│   └── ...
└── annotations/
    ├── 1_1_00001.json
    ├── 1_1_00002.json
    └── ...
```

### 2.3 데이터 분할
```python
from data.loader import split_dataset_files

split_dataset_files(
    source_dir="raw_data",
    output_dir="data",
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    seed=42
)
```

또는 Python 스크립트 실행:
```bash
python -c "from data.loader import split_dataset_files; split_dataset_files('raw_data', 'data')"
```

분할 후 디렉토리 구조:
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

## 3. 학습

### 3.1 기본 학습
```bash
python train.py \
    --data_root data \
    --experiment_name exp_001
```

### 3.2 백본 동결 학습 (빠른 학습)
```bash
python train.py \
    --data_root data \
    --experiment_name exp_freeze \
    --freeze_backbone
```

### 3.3 학습 재개
```bash
python train.py \
    --data_root data \
    --experiment_name exp_001 \
    --resume
```

### 3.4 학습 모니터링

#### TensorBoard
```bash
tensorboard --logdir logs/tensorboard
```
브라우저에서 http://localhost:6006 접속

#### 로그 파일 확인
```bash
tail -f logs/exp_001/exp_001.log
```

## 4. 추론

### 4.1 단일 이미지 추론
```bash
python inference.py \
    --checkpoint checkpoints/best_model.pth \
    --image data/test/images/1_1_00001.png \
    --output prediction.png \
    --score_threshold 0.5
```

### 4.2 여러 이미지 추론
```bash
python inference.py \
    --checkpoint checkpoints/best_model.pth \
    --input_dir data/test/images \
    --output predictions \
    --score_threshold 0.5
```

### 4.3 추론 결과
- 검출된 객체의 바운딩 박스
- 세그멘테이션 마스크
- 클래스 라벨 및 신뢰도 점수

## 5. 평가

### 5.1 테스트 데이터 평가
```bash
python evaluate.py \
    --checkpoint checkpoints/best_model.pth \
    --data_root data \
    --output evaluation_results.json
```

### 5.2 다양한 IoU 임계값 평가
```bash
python evaluate.py \
    --checkpoint checkpoints/best_model.pth \
    --data_root data \
    --iou_thresholds 0.5 0.75 0.9 \
    --output evaluation_results.json
```

### 5.3 평가 메트릭
- mAP (mean Average Precision)
- 클래스별 AP
- Precision, Recall, F1 Score

## 6. 설정 커스터마이징

### 6.1 config/config.py 수정
```python
# 학습 하이퍼파라미터
BATCH_SIZE = 4
LEARNING_RATE = 0.001
EPOCHS = 100

# 이미지 크기
INPUT_SIZE = (1024, 1024)

# 데이터 증강
AUGMENTATION = True
AUG_ROTATION = True
AUG_ROTATION_LIMIT = 15
```

### 6.2 config/model_config.py 수정
```python
# Mask R-CNN 모델 설정
backbone_name = "resnet50"
score_threshold = 0.5
nms_threshold = 0.5
```

## 7. 프로그래밍 방식 사용

### 7.1 학습 코드 예제
```python
from config.config import Config
from data.loader import get_data_loaders
from models.maskrcnn import create_model
import torch.optim as optim

# 설정
config = Config()

# 데이터 로더
train_loader, val_loader, test_loader = get_data_loaders(
    data_root="data",
    config=config
)

# 모델
model = create_model(
    num_classes=config.NUM_CLASSES,
    pretrained=True
)

# 옵티마이저
optimizer = optim.SGD(
    model.parameters(),
    lr=config.LEARNING_RATE,
    momentum=config.MOMENTUM
)

# 학습 루프
for epoch in range(config.EPOCHS):
    for images, targets in train_loader:
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
```

### 7.2 추론 코드 예제
```python
from inference import Inferencer

# Inferencer 생성
inferencer = Inferencer(
    checkpoint_path="checkpoints/best_model.pth",
    score_threshold=0.5
)

# 추론
prediction, image = inferencer.predict("test_image.png")

# 결과 저장
inferencer.predict_and_save("test_image.png", "result.png")
```

## 8. 문제 해결

### 8.1 CUDA Out of Memory
```python
# config/config.py
BATCH_SIZE = 2  # 배치 크기 줄이기
INPUT_SIZE = (512, 512)  # 이미지 크기 줄이기
```

### 8.2 학습 속도 개선
```python
# 백본 동결
python train.py --freeze_backbone

# Mixed Precision Training
# config/config.py
MIXED_PRECISION = True
```

### 8.3 과적합 방지
```python
# config/config.py
AUGMENTATION = True
EARLY_STOPPING_PATIENCE = 15
WEIGHT_DECAY = 0.0005
```

## 9. 베스트 프랙티스

### 9.1 학습 전 체크리스트
- [ ] 데이터가 올바르게 분할되었는지 확인
- [ ] 설정 파일 검토
- [ ] TensorBoard 준비
- [ ] 충분한 디스크 공간 확인

### 9.2 실험 관리
- 실험마다 고유한 이름 사용
- 하이퍼파라미터 기록
- 최고 성능 모델 별도 백업

### 9.3 성능 최적화
1. 작은 데이터셋으로 먼저 테스트
2. Learning Rate Finder 사용 고려
3. 데이터 증강 파라미터 조정
4. 앙상블 기법 적용

## 10. 추가 리소스

- [Mask R-CNN Paper](https://arxiv.org/abs/1703.06870)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [AI Hub 데이터셋](https://aihub.or.kr)
- [GitHub Issues](https://github.com/your-repo/issues)

## 문의사항
프로젝트 관련 문의는 GitHub Issues를 통해 남겨주세요.
