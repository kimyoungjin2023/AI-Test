"""
프로젝트 메인 설정 파일
AI Hub 내시경 이미지 데이터셋에 대한 전역 설정을 정의합니다.
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple


class Config:
    """전역 설정 클래스"""
    
    # ============ 프로젝트 경로 ============
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_ROOT = PROJECT_ROOT / "data"
    CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
    LOG_DIR = PROJECT_ROOT / "logs"
    OUTPUT_DIR = PROJECT_ROOT / "outputs"
    
    # ============ 데이터셋 정보 ============
    # 클래스 정의 (배경 포함)
    CLASSES = {
        0: "background",
        1: "stomach_ulcer",      # 위 궤양
        2: "stomach_polyp",       # 위 용종
        3: "stomach_cancer",      # 위 암
        4: "colon_ulcer",         # 대장 궤양
        5: "colon_polyp",         # 대장 용종
        6: "colon_cancer"         # 대장 암
    }
    
    NUM_CLASSES = len(CLASSES)  # 배경 포함 7개
    
    # 장기 타입
    ORGAN_TYPES = {
        0: "stomach",  # 위
        1: "colon"     # 대장
    }
    
    # 병변 타입
    LESION_TYPES = {
        0: "ulcer",    # 궤양
        1: "polyp",    # 용종
        2: "cancer"    # 암
    }
    
    # 색상 맵 (시각화용 - BGR 포맷)
    COLOR_MAP = {
        "background": (0, 0, 0),
        "stomach_ulcer": (255, 0, 0),      # Blue
        "stomach_polyp": (0, 255, 0),      # Green
        "stomach_cancer": (0, 0, 255),     # Red
        "colon_ulcer": (255, 255, 0),      # Cyan
        "colon_polyp": (255, 0, 255),      # Magenta
        "colon_cancer": (0, 255, 255)      # Yellow
    }
    
    # ============ 이미지 설정 ============
    IMAGE_SIZE = (2048, 2048)  # 원본 이미지 크기
    INPUT_SIZE = (1024, 1024)  # 모델 입력 크기 (메모리 효율성)
    
    # 이미지 정규화 파라미터 (ImageNet 기준)
    IMAGE_MEAN = [0.485, 0.456, 0.406]
    IMAGE_STD = [0.229, 0.224, 0.225]
    
    # ============ 학습 설정 ============
    BATCH_SIZE = 4
    NUM_WORKERS = 4
    EPOCHS = 100
    
    # Learning Rate
    LEARNING_RATE = 0.001
    LR_SCHEDULER = "step"  # Options: "step", "cosine", "plateau"
    LR_STEP_SIZE = 30
    LR_GAMMA = 0.1
    
    # Optimizer
    OPTIMIZER = "SGD"  # Options: "SGD", "Adam", "AdamW"
    MOMENTUM = 0.9
    WEIGHT_DECAY = 0.0001
    
    # ============ 데이터 분할 ============
    TRAIN_RATIO = 0.8
    VAL_RATIO = 0.1
    TEST_RATIO = 0.1
    
    # ============ 모델 설정 ============
    BACKBONE = "resnet50"  # Options: "resnet50", "resnet101"
    PRETRAINED = True  # COCO 사전학습 가중치 사용
    
    # RPN (Region Proposal Network) 설정
    RPN_ANCHOR_SIZES = ((32,), (64,), (128,), (256,), (512,))
    RPN_ASPECT_RATIOS = ((0.5, 1.0, 2.0),) * len(RPN_ANCHOR_SIZES)
    
    # ROI (Region of Interest) 설정
    ROI_BATCH_SIZE_PER_IMAGE = 512
    ROI_POSITIVE_FRACTION = 0.25
    
    # Detection 임계값
    SCORE_THRESHOLD = 0.5
    NMS_THRESHOLD = 0.5
    
    # ============ 데이터 증강 ============
    AUGMENTATION = True
    AUG_HORIZONTAL_FLIP = True
    AUG_VERTICAL_FLIP = True
    AUG_ROTATION = True
    AUG_ROTATION_LIMIT = 15
    AUG_BRIGHTNESS = True
    AUG_CONTRAST = True
    AUG_HUE_SATURATION = True
    
    # ============ 체크포인트 및 로깅 ============
    SAVE_CHECKPOINT_EVERY = 5  # 에폭마다 저장
    EARLY_STOPPING_PATIENCE = 15
    LOG_INTERVAL = 10  # 배치마다 로그 출력
    
    # TensorBoard
    USE_TENSORBOARD = True
    TENSORBOARD_LOG_DIR = LOG_DIR / "tensorboard"
    
    # Weights & Biases (선택사항)
    USE_WANDB = False
    WANDB_PROJECT = "endoscopy-maskrcnn"
    WANDB_ENTITY = None  # 사용자 계정명
    
    # ============ 평가 메트릭 ============
    IOU_THRESHOLDS = [0.5, 0.75, 0.9]
    EVAL_INTERVAL = 1  # 에폭마다 평가
    
    # ============ 추론 설정 ============
    INFERENCE_BATCH_SIZE = 1
    VISUALIZATION = True
    SAVE_PREDICTIONS = True
    
    # ============ 하드웨어 설정 ============
    DEVICE = "cuda"  # Options: "cuda", "cpu"
    MIXED_PRECISION = True  # FP16 학습
    
    # ============ 재현성 ============
    SEED = 42
    DETERMINISTIC = True
    
    @classmethod
    def create_directories(cls):
        """필요한 디렉토리 생성"""
        dirs = [
            cls.DATA_ROOT,
            cls.CHECKPOINT_DIR,
            cls.LOG_DIR,
            cls.OUTPUT_DIR,
            cls.TENSORBOARD_LOG_DIR
        ]
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_class_name(cls, class_id: int) -> str:
        """클래스 ID로부터 클래스 이름 반환"""
        return cls.CLASSES.get(class_id, "unknown")
    
    @classmethod
    def get_class_id(cls, organ: int, lesion: int) -> int:
        """장기 타입과 병변 타입으로부터 클래스 ID 계산"""
        # organ: 0(위), 1(대장)
        # lesion: 0(궤양), 1(용종), 2(암)
        return organ * 3 + lesion + 1  # +1은 배경(0) 때문
    
    @classmethod
    def get_color(cls, class_name: str) -> Tuple[int, int, int]:
        """클래스 이름으로부터 색상 반환"""
        return cls.COLOR_MAP.get(class_name, (128, 128, 128))
    
    @classmethod
    def print_config(cls):
        """현재 설정 출력"""
        print("=" * 60)
        print("Current Configuration")
        print("=" * 60)
        for attr_name in dir(cls):
            if not attr_name.startswith("_") and attr_name.isupper():
                attr_value = getattr(cls, attr_name)
                if not callable(attr_value):
                    print(f"{attr_name}: {attr_value}")
        print("=" * 60)


# 설정 검증
def validate_config():
    """설정값 검증"""
    assert Config.TRAIN_RATIO + Config.VAL_RATIO + Config.TEST_RATIO == 1.0, \
        "데이터 분할 비율의 합이 1.0이 아닙니다."
    
    assert Config.NUM_CLASSES == len(Config.CLASSES), \
        "클래스 개수가 일치하지 않습니다."
    
    assert Config.BATCH_SIZE > 0, "배치 크기는 0보다 커야 합니다."
    
    assert Config.LEARNING_RATE > 0, "학습률은 0보다 커야 합니다."
    
    print("✓ 설정 검증 완료")


if __name__ == "__main__":
    # 설정 출력 및 검증
    Config.print_config()
    validate_config()
    Config.create_directories()
    print("✓ 필요한 디렉토리 생성 완료")
