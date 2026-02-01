"""
Mask R-CNN 모델 하이퍼파라미터 설정
Detectron2 기반 설정 구조를 따릅니다.
"""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class MaskRCNNConfig:
    """Mask R-CNN 모델 설정"""
    
    # ============ Backbone ============
    backbone_name: str = "resnet50"  # resnet50, resnet101
    backbone_pretrained: bool = True
    backbone_freeze_at: int = 2  # 처음 N개 stage 동결 (0: 동결 안함)
    
    # ============ FPN (Feature Pyramid Network) ============
    fpn_out_channels: int = 256
    fpn_in_channels: List[int] = field(default_factory=lambda: [256, 512, 1024, 2048])
    
    # ============ RPN (Region Proposal Network) ============
    rpn_anchor_sizes: Tuple = ((32,), (64,), (128,), (256,), (512,))
    rpn_aspect_ratios: Tuple = ((0.5, 1.0, 2.0),)
    rpn_pre_nms_top_n_train: int = 2000
    rpn_pre_nms_top_n_test: int = 1000
    rpn_post_nms_top_n_train: int = 2000
    rpn_post_nms_top_n_test: int = 1000
    rpn_nms_thresh: float = 0.7
    rpn_fg_iou_thresh: float = 0.7  # Foreground IoU threshold
    rpn_bg_iou_thresh: float = 0.3  # Background IoU threshold
    rpn_batch_size_per_image: int = 256
    rpn_positive_fraction: float = 0.5
    
    # ============ ROI Heads ============
    # ROI Pooling
    roi_pooler_output_size: int = 7
    roi_pooler_sampling_ratio: int = 2
    roi_pooler_type: str = "ROIAlign"  # ROIAlign or ROIPool
    
    # Box Head
    box_head_fc_layers: List[int] = field(default_factory=lambda: [1024, 1024])
    box_predictor_smooth_l1_beta: float = 0.0
    
    # ROI Box Settings
    box_batch_size_per_image: int = 512
    box_positive_fraction: float = 0.25
    box_fg_iou_thresh: float = 0.5
    box_bg_iou_thresh: float = 0.5
    box_nms_thresh: float = 0.5
    box_score_thresh: float = 0.05  # 추론 시 점수 임계값
    box_detections_per_img: int = 100  # 이미지당 최대 검출 개수
    
    # ============ Mask Head ============
    mask_head_num_conv: int = 4
    mask_head_conv_dim: int = 256
    mask_pooler_resolution: int = 14
    mask_pooler_sampling_ratio: int = 2
    
    # ============ Loss Weights ============
    rpn_bbox_loss_weight: float = 1.0
    rpn_objectness_loss_weight: float = 1.0
    box_reg_loss_weight: float = 1.0
    box_cls_loss_weight: float = 1.0
    mask_loss_weight: float = 1.0
    
    # ============ Training ============
    # Image preprocessing
    pixel_mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    pixel_std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    
    # Input size
    min_size_train: int = 1024
    max_size_train: int = 1024
    min_size_test: int = 1024
    max_size_test: int = 1024
    
    def __post_init__(self):
        """설정 검증"""
        assert self.backbone_name in ["resnet50", "resnet101"], \
            f"지원하지 않는 백본: {self.backbone_name}"
        
        assert 0 <= self.box_positive_fraction <= 1, \
            "box_positive_fraction은 0과 1 사이여야 합니다."
        
        assert self.box_fg_iou_thresh >= self.box_bg_iou_thresh, \
            "Foreground IoU 임계값이 Background IoU 임계값보다 작습니다."


@dataclass
class OptimizerConfig:
    """옵티마이저 설정"""
    
    name: str = "SGD"  # SGD, Adam, AdamW
    base_lr: float = 0.001
    momentum: float = 0.9
    weight_decay: float = 0.0001
    
    # Learning Rate Scheduler
    scheduler_name: str = "step"  # step, cosine, plateau
    step_size: int = 30
    gamma: float = 0.1
    warmup_iters: int = 500
    warmup_factor: float = 0.001
    
    # For Cosine Annealing
    t_max: int = 100
    eta_min: float = 0.0
    
    # For ReduceLROnPlateau
    patience: int = 10
    factor: float = 0.1


@dataclass
class DataLoaderConfig:
    """데이터로더 설정"""
    
    batch_size: int = 4
    num_workers: int = 4
    shuffle_train: bool = True
    shuffle_val: bool = False
    pin_memory: bool = True
    drop_last: bool = True
    
    # Data Split
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    
    def __post_init__(self):
        """설정 검증"""
        assert abs(self.train_ratio + self.val_ratio + self.test_ratio - 1.0) < 1e-6, \
            "데이터 분할 비율의 합이 1.0이 아닙니다."


@dataclass
class AugmentationConfig:
    """데이터 증강 설정"""
    
    enabled: bool = True
    
    # Geometric Transformations
    horizontal_flip: bool = True
    horizontal_flip_prob: float = 0.5
    
    vertical_flip: bool = True
    vertical_flip_prob: float = 0.5
    
    rotation: bool = True
    rotation_limit: int = 15
    rotation_prob: float = 0.5
    
    # Color Transformations
    brightness_contrast: bool = True
    brightness_limit: float = 0.2
    contrast_limit: float = 0.2
    brightness_contrast_prob: float = 0.5
    
    hue_saturation: bool = True
    hue_shift_limit: int = 20
    sat_shift_limit: int = 30
    val_shift_limit: int = 20
    hue_saturation_prob: float = 0.5
    
    # Noise
    gaussian_noise: bool = False
    gaussian_noise_var_limit: Tuple[float, float] = (10.0, 50.0)
    gaussian_noise_prob: float = 0.3
    
    # Blur
    blur: bool = False
    blur_limit: int = 7
    blur_prob: float = 0.3


def get_default_configs():
    """기본 설정 딕셔너리 반환"""
    return {
        'model': MaskRCNNConfig(),
        'optimizer': OptimizerConfig(),
        'dataloader': DataLoaderConfig(),
        'augmentation': AugmentationConfig()
    }


if __name__ == "__main__":
    # 설정 테스트
    configs = get_default_configs()
    
    print("=" * 60)
    print("Model Configuration")
    print("=" * 60)
    for key, value in vars(configs['model']).items():
        print(f"{key}: {value}")
    
    print("\n" + "=" * 60)
    print("Optimizer Configuration")
    print("=" * 60)
    for key, value in vars(configs['optimizer']).items():
        print(f"{key}: {value}")
    
    print("\n✓ 모델 설정 검증 완료")
