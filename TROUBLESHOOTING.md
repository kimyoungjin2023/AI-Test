# 프로젝트 체크리스트 및 문제 해결 가이드

## ✅ 설치 및 설정 체크리스트

### 1단계: 환경 준비
- [ ] Python 3.8 이상 설치 확인
- [ ] CUDA 설치 확인 (GPU 사용 시)
- [ ] 가상환경 생성 및 활성화

```bash
python --version  # Python 3.8+ 확인
nvidia-smi        # CUDA 확인 (선택사항)
python -m venv venv
source venv/bin/activate  # Linux/Mac
```

### 2단계: 패키지 설치
- [ ] requirements.txt로 패키지 설치
- [ ] PyTorch 설치 확인
- [ ] 검증 스크립트 실행

```bash
pip install -r requirements.txt
python verify_project.py
```

### 3단계: 데이터 준비
- [ ] AI Hub에서 데이터 다운로드
- [ ] 데이터 압축 해제
- [ ] 데이터 분할 실행

```bash
python -c "from data.loader import split_dataset_files; split_dataset_files('raw_data', 'data')"
```

### 4단계: 학습 테스트
- [ ] 작은 데이터셋으로 학습 테스트
- [ ] TensorBoard 실행 확인
- [ ] 체크포인트 저장 확인

```bash
python train.py --data_root data --experiment_name test
tensorboard --logdir logs/tensorboard
```

## 🔧 일반적인 문제 해결

### 문제 1: "ModuleNotFoundError: No module named 'torch'"

**원인**: PyTorch가 설치되지 않음

**해결방법**:
```bash
# CPU 버전
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# GPU 버전 (CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# GPU 버전 (CUDA 12.1)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 문제 2: "CUDA out of memory"

**원인**: GPU 메모리 부족

**해결방법**:
```python
# config/config.py 수정
BATCH_SIZE = 2  # 4에서 2로 감소
INPUT_SIZE = (512, 512)  # 1024에서 512로 감소
```

또는 CPU 사용:
```python
DEVICE = "cpu"
```

### 문제 3: "FileNotFoundError: 이미지를 찾을 수 없습니다"

**원인**: 데이터 경로가 올바르지 않음

**해결방법**:
```bash
# 데이터 구조 확인
ls -R data/

# 올바른 구조:
# data/
# ├── train/
# │   ├── images/
# │   └── annotations/
# ├── val/
# │   ├── images/
# │   └── annotations/
# └── test/
#     ├── images/
#     └── annotations/
```

### 문제 4: JSON 파싱 오류

**원인**: 어노테이션 파일 형식이 예상과 다름

**확인사항**:
```bash
# JSON 파일 확인
cat data/train/annotations/1_1_00001.json | python -m json.tool

# 필수 필드 확인:
# - shapes (리스트)
# - shapes[i].organ (0 또는 1)
# - shapes[i].lesion (0, 1, 또는 2)
# - shapes[i].points (좌표 리스트)
# - shapes[i].shape_type ("polygon" 또는 "rectangle")
```

### 문제 5: "RuntimeError: Expected all tensors to be on the same device"

**원인**: 데이터와 모델이 다른 디바이스에 있음

**해결방법**:
이미 코드에서 처리되어 있지만, 확인:
```python
# train.py에서
images = [img.to(device) for img in images]
targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
```

### 문제 6: 학습 속도가 너무 느림

**해결방법**:

1. **백본 동결** (전이 학습):
```bash
python train.py --freeze_backbone
```

2. **워커 수 증가**:
```python
# config/config.py
NUM_WORKERS = 8  # CPU 코어 수에 맞게 조정
```

3. **Mixed Precision 활성화**:
```python
# config/config.py
MIXED_PRECISION = True
```

### 문제 7: 과적합 (Overfitting)

**증상**: 학습 loss는 감소하지만 검증 loss는 증가

**해결방법**:
```python
# config/config.py
AUGMENTATION = True  # 데이터 증강 활성화
WEIGHT_DECAY = 0.0005  # 정규화 강화
EARLY_STOPPING_PATIENCE = 10  # 조기 종료

# data/transforms.py에서 증강 강도 조정
AUG_ROTATION_LIMIT = 20
AUG_BRIGHTNESS_LIMIT = 0.3
```

### 문제 8: TensorBoard가 실행되지 않음

**해결방법**:
```bash
# TensorBoard 설치 확인
pip install tensorboard

# 포트 변경하여 실행
tensorboard --logdir logs/tensorboard --port 6007

# 브라우저에서 접속
http://localhost:6007
```

## 📊 성능 최적화 팁

### 1. Learning Rate 튜닝
```python
# config/config.py
LEARNING_RATE = 0.001  # 기본값
# 너무 높으면: 학습 불안정
# 너무 낮으면: 학습 속도 느림

# 권장: Learning Rate Finder 사용
```

### 2. Batch Size 조정
```python
# GPU 메모리에 따라 조정
# RTX 3090 (24GB): BATCH_SIZE = 8
# RTX 3080 (10GB): BATCH_SIZE = 4
# RTX 3060 (12GB): BATCH_SIZE = 4-6
```

### 3. 데이터 증강 최적화
```python
# 의료 이미지 특성에 맞게
AUG_ROTATION = True  # 회전
AUG_HORIZONTAL_FLIP = True  # 좌우 반전
AUG_VERTICAL_FLIP = True  # 상하 반전
AUG_BRIGHTNESS = True  # 밝기 (내시경 조명)
AUG_HUE_SATURATION = True  # 색조 (조직 색상)
```

### 4. 앙상블 기법
```python
# 여러 모델 학습 후 평균
# 1. 다른 시드로 여러 번 학습
# 2. 다른 백본 사용 (ResNet50, ResNet101)
# 3. 예측 결과 평균 또는 투표
```

## 🐛 디버깅 팁

### 로그 확인
```bash
# 학습 로그
tail -f logs/exp_001/exp_001.log

# TensorBoard
tensorboard --logdir logs/tensorboard
```

### 데이터 샘플 확인
```python
from data.dataset import EndoscopyDataset
from data.transforms import get_train_transforms

dataset = EndoscopyDataset('data', 'train', get_train_transforms())
image, target = dataset[0]

print(f"Image shape: {image.shape}")
print(f"Num boxes: {len(target['boxes'])}")
print(f"Labels: {target['labels']}")
```

### 모델 출력 확인
```python
model.eval()
with torch.no_grad():
    outputs = model([image.to(device)])
    print(f"Predictions: {len(outputs[0]['boxes'])}")
```

## 📝 코드 검증

### 전체 검증
```bash
python verify_project.py
```

### 개별 모듈 테스트
```bash
# Config 테스트
python config/config.py

# Dataset 테스트
python data/dataset.py

# Model 테스트
python models/maskrcnn.py
```

## 💡 베스트 프랙티스

1. **실험 관리**
   - 각 실험마다 고유한 이름 사용
   - 하이퍼파라미터를 로그에 기록
   - 최고 성능 모델 별도 백업

2. **버전 관리**
   - Git으로 코드 관리
   - .gitignore에 data/, logs/, checkpoints/ 추가

3. **재현성**
   - 시드 고정 (SEED = 42)
   - 설정 파일 저장
   - 환경 정보 기록

4. **점진적 개선**
   - 작은 데이터셋으로 먼저 테스트
   - 단계별로 복잡도 증가
   - 각 변경사항의 영향 측정

## 📞 추가 지원

문제가 해결되지 않으면:
1. verify_project.py 실행 결과 확인
2. 오류 메시지 전체 복사
3. 사용 중인 환경 정보 (OS, Python 버전, GPU 등)
4. GitHub Issues에 문의

## ✨ 성공적인 학습을 위한 체크리스트

- [ ] GPU 메모리 충분 (최소 8GB 권장)
- [ ] 데이터가 올바르게 로드됨
- [ ] 첫 에폭이 정상적으로 완료됨
- [ ] Loss가 감소하는 추세
- [ ] TensorBoard에서 시각화 확인
- [ ] 체크포인트가 정상 저장됨
- [ ] 검증 mAP가 개선되는 추세
