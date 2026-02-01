"""
데이터 증강 변환
Albumentations 라이브러리를 사용한 의료 이미지 증강
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
from typing import Optional

from config.config import Config
from config.model_config import AugmentationConfig


class EndoscopyTransforms:
    """내시경 이미지 증강 변환 클래스"""
    
    def __init__(
        self,
        config: Optional[Config] = None,
        aug_config: Optional[AugmentationConfig] = None,
        is_train: bool = True
    ):
        """
        Args:
            config: 전역 설정
            aug_config: 증강 설정
            is_train: 학습 모드 여부
        """
        self.config = config or Config()
        self.aug_config = aug_config or AugmentationConfig()
        self.is_train = is_train
        
        self.transform = self._build_transform()
    
    def _build_transform(self) -> A.Compose:
        """증강 파이프라인 생성"""
        
        transforms_list = []
        
        # 학습 모드에서만 증강 적용
        if self.is_train and self.aug_config.enabled:
            # Geometric Transformations
            if self.aug_config.horizontal_flip:
                transforms_list.append(
                    A.HorizontalFlip(p=self.aug_config.horizontal_flip_prob)
                )
            
            if self.aug_config.vertical_flip:
                transforms_list.append(
                    A.VerticalFlip(p=self.aug_config.vertical_flip_prob)
                )
            
            if self.aug_config.rotation:
                transforms_list.append(
                    A.Rotate(
                        limit=self.aug_config.rotation_limit,
                        interpolation=cv2.INTER_LINEAR,
                        border_mode=cv2.BORDER_CONSTANT,
                        value=0,
                        p=self.aug_config.rotation_prob
                    )
                )
            
            # Color Transformations
            if self.aug_config.brightness_contrast:
                transforms_list.append(
                    A.RandomBrightnessContrast(
                        brightness_limit=self.aug_config.brightness_limit,
                        contrast_limit=self.aug_config.contrast_limit,
                        p=self.aug_config.brightness_contrast_prob
                    )
                )
            
            if self.aug_config.hue_saturation:
                transforms_list.append(
                    A.HueSaturationValue(
                        hue_shift_limit=self.aug_config.hue_shift_limit,
                        sat_shift_limit=self.aug_config.sat_shift_limit,
                        val_shift_limit=self.aug_config.val_shift_limit,
                        p=self.aug_config.hue_saturation_prob
                    )
                )
            
            # Noise (선택적)
            if self.aug_config.gaussian_noise:
                transforms_list.append(
                    A.GaussNoise(
                        var_limit=self.aug_config.gaussian_noise_var_limit,
                        p=self.aug_config.gaussian_noise_prob
                    )
                )
            
            # Blur (선택적)
            if self.aug_config.blur:
                transforms_list.append(
                    A.Blur(
                        blur_limit=self.aug_config.blur_limit,
                        p=self.aug_config.blur_prob
                    )
                )
        
        # Resize (항상 적용)
        transforms_list.append(
            A.Resize(
                height=self.config.INPUT_SIZE[0],
                width=self.config.INPUT_SIZE[1],
                interpolation=cv2.INTER_LINEAR
            )
        )
        
        # Compose 생성 - bbox_params 추가
        transform = A.Compose(
            transforms_list,
            bbox_params=A.BboxParams(
                format='pascal_voc',  # [x_min, y_min, x_max, y_max]
                label_fields=['labels'],
                min_area=0,
                min_visibility=0.3  # 30% 이상 보이는 박스만 유지
            )
        )
        
        return transform
    
    def __call__(self, **kwargs):
        """변환 적용"""
        return self.transform(**kwargs)


def get_train_transforms(config: Optional[Config] = None) -> EndoscopyTransforms:
    """학습용 변환 반환"""
    return EndoscopyTransforms(config=config, is_train=True)


def get_val_transforms(config: Optional[Config] = None) -> EndoscopyTransforms:
    """검증용 변환 반환"""
    aug_config = AugmentationConfig(enabled=False)  # 증강 비활성화
    return EndoscopyTransforms(config=config, aug_config=aug_config, is_train=False)


def get_test_transforms(config: Optional[Config] = None) -> EndoscopyTransforms:
    """테스트용 변환 반환"""
    aug_config = AugmentationConfig(enabled=False)  # 증강 비활성화
    return EndoscopyTransforms(config=config, aug_config=aug_config, is_train=False)


class MedicalImageAugmentation:
    """
    의료 이미지 특화 증강 클래스
    내시경 이미지 특성을 고려한 추가 증강 기법
    """
    
    @staticmethod
    def get_advanced_transforms(config: Optional[Config] = None) -> A.Compose:
        """
        고급 의료 이미지 증강
        
        의료 이미지 특성:
        - 조명 변화 (내시경 광원)
        - 조직의 색상 변화
        - 촬영 각도 변화
        """
        config = config or Config()
        
        transforms = A.Compose([
            # 1. 기하학적 변환
            A.OneOf([
                A.HorizontalFlip(p=1.0),
                A.VerticalFlip(p=1.0),
                A.Rotate(limit=20, p=1.0),
            ], p=0.7),
            
            # 2. 조명 및 색상 변화 (내시경 특성)
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=0.3,
                    contrast_limit=0.3,
                    p=1.0
                ),
                A.HueSaturationValue(
                    hue_shift_limit=15,
                    sat_shift_limit=25,
                    val_shift_limit=15,
                    p=1.0
                ),
                A.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1,
                    p=1.0
                ),
            ], p=0.8),
            
            # 3. 노이즈 및 블러 (이미지 품질 시뮬레이션)
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 30.0), p=1.0),
                A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                A.MotionBlur(blur_limit=5, p=1.0),
            ], p=0.3),
            
            # 4. 세부 조정
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
            A.Sharpen(alpha=(0.1, 0.3), lightness=(0.8, 1.2), p=0.2),
            
            # 5. Resize
            A.Resize(
                height=config.INPUT_SIZE[0],
                width=config.INPUT_SIZE[1]
            ),
        ],
        bbox_params=A.BboxParams(
            format='pascal_voc',
            label_fields=['labels'],
            min_area=0,
            min_visibility=0.3
        ))
        
        return transforms
    
    @staticmethod
    def get_light_transforms() -> A.Compose:
        """경량 증강 (빠른 학습용)"""
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.Resize(height=1024, width=1024),
        ],
        bbox_params=A.BboxParams(
            format='pascal_voc',
            label_fields=['labels'],
            min_area=0,
            min_visibility=0.3
        ))


def visualize_augmentation(image: np.ndarray, transform: A.Compose, num_samples: int = 5):
    """
    증강 결과 시각화 (디버깅용)
    
    Args:
        image: 원본 이미지 (H, W, C)
        transform: Albumentations 변환
        num_samples: 생성할 증강 샘플 수
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, num_samples + 1, figsize=(20, 4))
    
    # 원본 이미지
    axes[0].imshow(image)
    axes[0].set_title("Original")
    axes[0].axis('off')
    
    # 증강 샘플들
    for i in range(num_samples):
        # 더미 박스와 라벨 (변환 적용을 위해)
        transformed = transform(
            image=image,
            bboxes=[[100, 100, 200, 200]],
            labels=[1]
        )
        aug_image = transformed['image']
        
        axes[i + 1].imshow(aug_image)
        axes[i + 1].set_title(f"Augmented {i+1}")
        axes[i + 1].axis('off')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 테스트 코드
    print("EndoscopyTransforms 테스트")
    print("=" * 60)
    
    # 학습용 변환
    train_transform = get_train_transforms()
    print("✓ 학습용 변환 생성 완료")
    
    # 검증용 변환
    val_transform = get_val_transforms()
    print("✓ 검증용 변환 생성 완료")
    
    # 고급 의료 이미지 증강
    advanced_transform = MedicalImageAugmentation.get_advanced_transforms()
    print("✓ 고급 증강 변환 생성 완료")
    
    # 더미 이미지로 테스트
    dummy_image = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)
    dummy_boxes = [[100, 100, 300, 300], [400, 400, 600, 600]]
    dummy_labels = [1, 2]
    
    result = train_transform(
        image=dummy_image,
        bboxes=dummy_boxes,
        labels=dummy_labels
    )
    
    print(f"\n변환 후 이미지 shape: {result['image'].shape}")
    print(f"변환 후 박스 개수: {len(result['bboxes'])}")
    print("✓ 증강 테스트 완료")
