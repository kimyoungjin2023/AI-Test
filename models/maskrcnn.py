"""
Mask R-CNN 모델 정의
TorchVision의 Mask R-CNN을 사용하여 내시경 이미지 분할 모델 구현
"""

import torch
import torch.nn as nn
from torchvision.models.detection import maskrcnn_resnet50_fpn, maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from typing import Optional, Dict, List

from config.config import Config
from config.model_config import MaskRCNNConfig


class EndoscopyMaskRCNN(nn.Module):
    """
    내시경 이미지용 Mask R-CNN 모델
    """
    
    def __init__(
        self,
        num_classes: int,
        config: Optional[Config] = None,
        model_config: Optional[MaskRCNNConfig] = None,
        pretrained: bool = True
    ):
        """
        Args:
            num_classes: 클래스 개수 (배경 포함)
            config: 전역 설정
            model_config: 모델 설정
            pretrained: COCO 사전학습 가중치 사용 여부
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.config = config or Config()
        self.model_config = model_config or MaskRCNNConfig()
        
        # Mask R-CNN 모델 로드
        self.model = self._build_model(pretrained)
        
        print(f"✓ Mask R-CNN 모델 초기화 완료")
        print(f"  - 백본: {self.model_config.backbone_name}")
        print(f"  - 클래스 수: {num_classes}")
        print(f"  - 사전학습: {pretrained}")
    
    def _build_model(self, pretrained: bool) -> nn.Module:
        """
        Mask R-CNN 모델 빌드
        
        Args:
            pretrained: COCO 사전학습 가중치 사용 여부
            
        Returns:
            Mask R-CNN 모델
        """
        # 사전학습된 Mask R-CNN 로드
        if self.model_config.backbone_name == "resnet50":
            model = maskrcnn_resnet50_fpn(pretrained=pretrained)
        else:
            raise ValueError(f"지원하지 않는 백본: {self.model_config.backbone_name}")
        
        # Box Predictor 교체 (클래스 수 변경)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)
        
        # Mask Predictor 교체
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = self.model_config.mask_head_conv_dim
        model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask,
            hidden_layer,
            self.num_classes
        )
        
        return model
    
    def forward(
        self,
        images: List[torch.Tensor],
        targets: Optional[List[Dict[str, torch.Tensor]]] = None
    ):
        """
        Forward pass
        
        Args:
            images: List of tensors (C, H, W)
            targets: List of target dicts (학습 시에만 필요)
            
        Returns:
            학습 모드: loss dict
            추론 모드: predictions dict
        """
        return self.model(images, targets)
    
    def predict(
        self,
        images: List[torch.Tensor],
        score_threshold: float = 0.5
    ) -> List[Dict[str, torch.Tensor]]:
        """
        추론 수행
        
        Args:
            images: List of image tensors
            score_threshold: 검출 점수 임계값
            
        Returns:
            List of prediction dicts
        """
        self.model.eval()
        
        with torch.no_grad():
            predictions = self.model(images)
        
        # 점수 임계값 필터링
        filtered_predictions = []
        for pred in predictions:
            keep = pred['scores'] > score_threshold
            filtered_pred = {
                'boxes': pred['boxes'][keep],
                'labels': pred['labels'][keep],
                'scores': pred['scores'][keep],
                'masks': pred['masks'][keep]
            }
            filtered_predictions.append(filtered_pred)
        
        return filtered_predictions
    
    def freeze_backbone(self, freeze_at: int = 2):
        """
        백본 네트워크 동결
        
        Args:
            freeze_at: 동결할 stage 수 (0: 동결 안함)
        """
        if freeze_at <= 0:
            return
        
        # ResNet backbone의 layer 동결
        backbone = self.model.backbone.body
        
        # 초기 레이어 동결
        for name, parameter in backbone.named_parameters():
            if "layer" + str(freeze_at) not in name:
                parameter.requires_grad = False
        
        print(f"✓ 백본 네트워크 일부 동결 (freeze_at={freeze_at})")
    
    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """학습 가능한 파라미터 반환"""
        return [p for p in self.model.parameters() if p.requires_grad]
    
    def count_parameters(self) -> Dict[str, int]:
        """파라미터 개수 계산"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'total': total_params,
            'trainable': trainable_params,
            'frozen': total_params - trainable_params
        }


class MaskRCNNBuilder:
    """Mask R-CNN 모델 빌더 클래스"""
    
    def __init__(self):
        self.num_classes = None
        self.config = Config()
        self.model_config = MaskRCNNConfig()
        self.pretrained = True
        self.freeze_backbone = False
        self.freeze_at = 2
    
    def set_num_classes(self, num_classes: int):
        """클래스 수 설정"""
        self.num_classes = num_classes
        return self
    
    def set_config(self, config: Config):
        """전역 설정"""
        self.config = config
        return self
    
    def set_model_config(self, model_config: MaskRCNNConfig):
        """모델 설정"""
        self.model_config = model_config
        return self
    
    def set_pretrained(self, pretrained: bool):
        """사전학습 가중치 사용 여부"""
        self.pretrained = pretrained
        return self
    
    def set_freeze_backbone(self, freeze: bool, freeze_at: int = 2):
        """백본 동결 설정"""
        self.freeze_backbone = freeze
        self.freeze_at = freeze_at
        return self
    
    def build(self) -> EndoscopyMaskRCNN:
        """모델 빌드"""
        if self.num_classes is None:
            raise ValueError("num_classes가 설정되지 않았습니다.")
        
        model = EndoscopyMaskRCNN(
            num_classes=self.num_classes,
            config=self.config,
            model_config=self.model_config,
            pretrained=self.pretrained
        )
        
        if self.freeze_backbone:
            model.freeze_backbone(self.freeze_at)
        
        return model


def create_model(
    num_classes: int = 7,
    pretrained: bool = True,
    freeze_backbone: bool = False
) -> EndoscopyMaskRCNN:
    """
    편의 함수: Mask R-CNN 모델 생성
    
    Args:
        num_classes: 클래스 개수
        pretrained: COCO 사전학습 가중치 사용
        freeze_backbone: 백본 네트워크 동결
        
    Returns:
        EndoscopyMaskRCNN 모델
    """
    model = EndoscopyMaskRCNN(
        num_classes=num_classes,
        pretrained=pretrained
    )
    
    if freeze_backbone:
        model.freeze_backbone(freeze_at=2)
    
    return model


def print_model_summary(model: EndoscopyMaskRCNN):
    """모델 요약 정보 출력"""
    params = model.count_parameters()
    
    print("\n" + "=" * 60)
    print("모델 요약")
    print("=" * 60)
    print(f"총 파라미터: {params['total']:,}")
    print(f"학습 가능 파라미터: {params['trainable']:,}")
    print(f"동결된 파라미터: {params['frozen']:,}")
    print(f"클래스 수: {model.num_classes}")
    print("=" * 60)


if __name__ == "__main__":
    # 테스트 코드
    print("EndoscopyMaskRCNN 테스트")
    print("=" * 60)
    
    # 모델 생성
    num_classes = 7  # 배경 + 6개 병변 클래스
    model = create_model(num_classes=num_classes, pretrained=True)
    
    # 모델 요약
    print_model_summary(model)
    
    # 더미 데이터로 Forward pass 테스트
    print("\n더미 데이터로 Forward pass 테스트...")
    
    # 더미 입력
    dummy_images = [torch.rand(3, 1024, 1024) for _ in range(2)]
    
    # 더미 타겟 (학습 모드)
    dummy_targets = [
        {
            'boxes': torch.tensor([[100, 100, 300, 300]], dtype=torch.float32),
            'labels': torch.tensor([1], dtype=torch.int64),
            'masks': torch.zeros((1, 1024, 1024), dtype=torch.uint8),
            'image_id': torch.tensor([0]),
            'area': torch.tensor([40000.0]),
            'iscrowd': torch.tensor([0])
        },
        {
            'boxes': torch.tensor([[200, 200, 400, 400]], dtype=torch.float32),
            'labels': torch.tensor([2], dtype=torch.int64),
            'masks': torch.zeros((1, 1024, 1024), dtype=torch.uint8),
            'image_id': torch.tensor([1]),
            'area': torch.tensor([40000.0]),
            'iscrowd': torch.tensor([0])
        }
    ]
    
    # 학습 모드
    model.train()
    loss_dict = model(dummy_images, dummy_targets)
    print(f"\n학습 모드 Loss:")
    for key, value in loss_dict.items():
        print(f"  {key}: {value.item():.4f}")
    
    # 추론 모드
    predictions = model.predict(dummy_images, score_threshold=0.5)
    print(f"\n추론 모드 결과:")
    print(f"  배치 크기: {len(predictions)}")
    print(f"  첫 번째 이미지 검출 개수: {len(predictions[0]['boxes'])}")
    
    print("\n✓ 모델 테스트 완료")
