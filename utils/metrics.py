"""
평가 메트릭 계산
mAP, IoU, Precision, Recall 등
"""

import torch
import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict


def calculate_iou(box1: torch.Tensor, box2: torch.Tensor) -> float:
    """
    두 박스 간의 IoU 계산
    
    Args:
        box1, box2: [x1, y1, x2, y2] 형식의 박스
        
    Returns:
        IoU 값
    """
    # 교집합 영역 계산
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # 합집합 영역 계산
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def calculate_map(
    predictions: List[Dict],
    targets: List[Dict],
    iou_threshold: float = 0.5,
    num_classes: int = 7
) -> Dict[str, float]:
    """
    mAP (mean Average Precision) 계산
    
    Args:
        predictions: 예측 결과 리스트
        targets: 정답 타겟 리스트
        iou_threshold: IoU 임계값
        num_classes: 클래스 개수
        
    Returns:
        메트릭 딕셔너리
    """
    # 클래스별 AP 계산
    ap_per_class = {}
    
    for class_id in range(1, num_classes):  # 배경(0) 제외
        # 해당 클래스의 예측 및 정답 수집
        class_preds = []
        class_targets = []
        
        for pred, target in zip(predictions, targets):
            # 예측에서 해당 클래스 필터링
            class_mask = pred['labels'] == class_id
            if class_mask.any():
                class_preds.append({
                    'boxes': pred['boxes'][class_mask],
                    'scores': pred['scores'][class_mask]
                })
            else:
                class_preds.append({'boxes': torch.empty(0, 4), 'scores': torch.empty(0)})
            
            # 타겟에서 해당 클래스 필터링
            target_mask = target['labels'] == class_id
            class_targets.append(target['boxes'][target_mask])
        
        # AP 계산
        ap = calculate_ap(class_preds, class_targets, iou_threshold)
        ap_per_class[class_id] = ap
    
    # mAP 계산
    mean_ap = np.mean(list(ap_per_class.values()))
    
    return {
        'mAP': mean_ap,
        **{f'AP_class_{i}': ap for i, ap in ap_per_class.items()}
    }


def calculate_ap(predictions: List[Dict], targets: List[torch.Tensor], iou_threshold: float) -> float:
    """
    단일 클래스에 대한 Average Precision 계산
    """
    all_scores = []
    all_matched = []
    num_gt = 0
    
    for pred, gt in zip(predictions, targets):
        num_gt += len(gt)
        
        if len(pred['boxes']) == 0:
            continue
        
        # 점수로 정렬
        sorted_indices = torch.argsort(pred['scores'], descending=True)
        pred_boxes = pred['boxes'][sorted_indices]
        pred_scores = pred['scores'][sorted_indices]
        
        matched = torch.zeros(len(pred_boxes), dtype=torch.bool)
        gt_matched = torch.zeros(len(gt), dtype=torch.bool)
        
        # 각 예측을 GT와 매칭
        for i, pred_box in enumerate(pred_boxes):
            best_iou = 0
            best_gt_idx = -1
            
            for j, gt_box in enumerate(gt):
                if gt_matched[j]:
                    continue
                
                iou = calculate_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j
            
            if best_iou >= iou_threshold:
                matched[i] = True
                if best_gt_idx >= 0:
                    gt_matched[best_gt_idx] = True
        
        all_scores.extend(pred_scores.cpu().numpy())
        all_matched.extend(matched.cpu().numpy())
    
    if num_gt == 0:
        return 0.0
    
    # Precision-Recall 커브 계산
    sorted_indices = np.argsort(all_scores)[::-1]
    matched_sorted = np.array(all_matched)[sorted_indices]
    
    tp = np.cumsum(matched_sorted)
    fp = np.cumsum(~matched_sorted)
    
    precision = tp / (tp + fp + 1e-10)
    recall = tp / num_gt
    
    # AP 계산 (11-point interpolation)
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        if np.sum(recall >= t) == 0:
            p = 0
        else:
            p = np.max(precision[recall >= t])
        ap += p / 11
    
    return ap


class ConfusionMatrix:
    """혼동 행렬 계산"""
    
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        """
        혼동 행렬 업데이트
        
        Args:
            predictions: 예측 라벨 (N,)
            targets: 정답 라벨 (N,)
        """
        for pred, target in zip(predictions.cpu().numpy(), targets.cpu().numpy()):
            self.matrix[target, pred] += 1
    
    def get_metrics(self) -> Dict[str, float]:
        """Precision, Recall, F1 계산"""
        tp = np.diag(self.matrix)
        fp = self.matrix.sum(axis=0) - tp
        fn = self.matrix.sum(axis=1) - tp
        
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
        
        return {
            'precision': precision.mean(),
            'recall': recall.mean(),
            'f1_score': f1.mean()
        }
    
    def reset(self):
        """혼동 행렬 초기화"""
        self.matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)


if __name__ == "__main__":
    print("Metrics 테스트")
    print("=" * 60)
    
    # IoU 테스트
    box1 = torch.tensor([100, 100, 200, 200])
    box2 = torch.tensor([150, 150, 250, 250])
    iou = calculate_iou(box1, box2)
    print(f"IoU: {iou:.4f}")
    
    # Confusion Matrix 테스트
    cm = ConfusionMatrix(num_classes=3)
    preds = torch.tensor([0, 1, 2, 1, 0])
    targets = torch.tensor([0, 1, 1, 2, 0])
    cm.update(preds, targets)
    metrics = cm.get_metrics()
    print(f"\n메트릭: {metrics}")
    
    print("\n✓ Metrics 테스트 완료")
