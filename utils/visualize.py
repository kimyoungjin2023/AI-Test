"""
시각화 유틸리티
검출 결과, 마스크, 박스 등을 시각화
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import List, Dict, Optional, Tuple
from pathlib import Path

from config.config import Config


class Visualizer:
    """검출 결과 시각화 클래스"""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
    
    def draw_boxes_and_masks(
        self,
        image: np.ndarray,
        boxes: torch.Tensor,
        labels: torch.Tensor,
        scores: Optional[torch.Tensor] = None,
        masks: Optional[torch.Tensor] = None,
        thickness: int = 2,
        alpha: float = 0.4
    ) -> np.ndarray:
        """
        이미지에 박스와 마스크 그리기
        
        Args:
            image: RGB 이미지 (H, W, 3)
            boxes: (N, 4) 박스 [x1, y1, x2, y2]
            labels: (N,) 클래스 라벨
            scores: (N,) 점수 (선택적)
            masks: (N, H, W) 마스크 (선택적)
            thickness: 박스 두께
            alpha: 마스크 투명도
            
        Returns:
            시각화된 이미지
        """
        # 이미지 복사
        vis_image = image.copy()
        
        # 텐서를 numpy로 변환
        if isinstance(boxes, torch.Tensor):
            boxes = boxes.cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
        if scores is not None and isinstance(scores, torch.Tensor):
            scores = scores.cpu().numpy()
        if masks is not None and isinstance(masks, torch.Tensor):
            masks = masks.cpu().numpy()
        
        # 각 객체에 대해 시각화
        for i in range(len(boxes)):
            box = boxes[i].astype(int)
            label = int(labels[i])
            score = scores[i] if scores is not None else None
            
            # 클래스 정보
            class_name = self.config.get_class_name(label)
            color = self.config.get_color(class_name)
            
            # 마스크 그리기 (있는 경우)
            if masks is not None and i < len(masks):
                mask = masks[i]
                if len(mask.shape) == 3:  # (1, H, W)
                    mask = mask[0]
                
                # 마스크를 이미지 크기로 리사이즈
                if mask.shape != vis_image.shape[:2]:
                    mask = cv2.resize(mask, (vis_image.shape[1], vis_image.shape[0]))
                
                # 마스크 적용
                mask_bool = mask > 0.5
                colored_mask = np.zeros_like(vis_image)
                colored_mask[mask_bool] = color
                vis_image = cv2.addWeighted(vis_image, 1.0, colored_mask, alpha, 0)
            
            # 박스 그리기
            cv2.rectangle(
                vis_image,
                (box[0], box[1]),
                (box[2], box[3]),
                color,
                thickness
            )
            
            # 라벨 텍스트
            text = class_name
            if score is not None:
                text += f" {score:.2f}"
            
            # 텍스트 배경
            (text_width, text_height), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                vis_image,
                (box[0], box[1] - text_height - 5),
                (box[0] + text_width, box[1]),
                color,
                -1
            )
            
            # 텍스트
            cv2.putText(
                vis_image,
                text,
                (box[0], box[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
        
        return vis_image
    
    def visualize_batch(
        self,
        images: torch.Tensor,
        predictions: List[Dict],
        save_dir: Optional[str] = None,
        prefix: str = "batch"
    ):
        """
        배치 이미지 시각화
        
        Args:
            images: (B, C, H, W) 이미지 텐서
            predictions: 예측 결과 리스트
            save_dir: 저장 디렉토리
            prefix: 파일명 접두사
        """
        batch_size = images.shape[0]
        
        fig, axes = plt.subplots(1, min(batch_size, 4), figsize=(20, 5))
        if batch_size == 1:
            axes = [axes]
        
        for i in range(min(batch_size, 4)):
            # 이미지 역정규화
            image = images[i].cpu().numpy().transpose(1, 2, 0)
            mean = np.array(self.config.IMAGE_MEAN)
            std = np.array(self.config.IMAGE_STD)
            image = (image * std + mean) * 255
            image = image.astype(np.uint8)
            
            # 예측 그리기
            pred = predictions[i]
            vis_image = self.draw_boxes_and_masks(
                image,
                pred['boxes'],
                pred['labels'],
                pred['scores'],
                pred.get('masks')
            )
            
            axes[i].imshow(vis_image)
            axes[i].set_title(f"Image {i+1}")
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_dir:
            save_path = Path(save_dir) / f"{prefix}.png"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_metrics(
        self,
        metrics_history: Dict[str, List[float]],
        save_path: Optional[str] = None
    ):
        """
        메트릭 그래프 플롯
        
        Args:
            metrics_history: {'metric_name': [values]}
            save_path: 저장 경로
        """
        num_metrics = len(metrics_history)
        fig, axes = plt.subplots(1, num_metrics, figsize=(6*num_metrics, 5))
        
        if num_metrics == 1:
            axes = [axes]
        
        for ax, (metric_name, values) in zip(axes, metrics_history.items()):
            ax.plot(values, marker='o')
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric_name)
            ax.set_title(f'{metric_name} over epochs')
            ax.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


def save_prediction_image(
    image_path: str,
    prediction: Dict,
    save_path: str,
    config: Optional[Config] = None
):
    """
    단일 이미지 예측 결과 저장
    
    Args:
        image_path: 원본 이미지 경로
        prediction: 예측 결과
        save_path: 저장 경로
        config: Config 객체
    """
    config = config or Config()
    visualizer = Visualizer(config)
    
    # 이미지 로드
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 시각화
    vis_image = visualizer.draw_boxes_and_masks(
        image,
        prediction['boxes'],
        prediction['labels'],
        prediction['scores'],
        prediction.get('masks')
    )
    
    # 저장
    vis_image_bgr = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, vis_image_bgr)


if __name__ == "__main__":
    print("Visualizer 테스트")
    print("=" * 60)
    
    # 더미 데이터 생성
    image = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)
    boxes = torch.tensor([[100, 100, 300, 300], [400, 400, 600, 600]])
    labels = torch.tensor([1, 2])
    scores = torch.tensor([0.9, 0.85])
    
    # Visualizer 생성
    visualizer = Visualizer()
    
    # 시각화
    vis_image = visualizer.draw_boxes_and_masks(image, boxes, labels, scores)
    print(f"시각화 이미지 shape: {vis_image.shape}")
    
    print("\n✓ Visualizer 테스트 완료")
