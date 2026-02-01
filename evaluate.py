"""
Mask R-CNN 평가 스크립트
테스트 데이터셋에 대한 정량적 평가
"""

import argparse
import torch
from tqdm import tqdm
import sys
from pathlib import Path
import json

sys.path.append(str(Path(__file__).parent))

from config.config import Config
from models.maskrcnn import create_model
from data.loader import get_data_loaders
from utils.metrics import calculate_map, ConfusionMatrix
from utils.checkpoint import CheckpointManager


@torch.no_grad()
def evaluate_model(
    model,
    data_loader,
    device,
    config,
    iou_thresholds=[0.5, 0.75, 0.9]
):
    """
    모델 평가
    
    Args:
        model: 모델
        data_loader: 데이터 로더
        device: 디바이스
        config: Config 객체
        iou_thresholds: IoU 임계값 리스트
        
    Returns:
        평가 메트릭 딕셔너리
    """
    model.eval()
    
    all_predictions = []
    all_targets = []
    confusion_matrix = ConfusionMatrix(num_classes=config.NUM_CLASSES)
    
    print("\n평가 진행 중...")
    for images, targets in tqdm(data_loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # 추론
        outputs = model(images)
        
        all_predictions.extend(outputs)
        all_targets.extend(targets)
        
        # Confusion Matrix 업데이트
        for pred, target in zip(outputs, targets):
            if len(pred['labels']) > 0 and len(target['labels']) > 0:
                confusion_matrix.update(pred['labels'], target['labels'])
    
    # 각 IoU 임계값에 대해 mAP 계산
    results = {}
    
    for iou_thresh in iou_thresholds:
        map_metrics = calculate_map(
            all_predictions,
            all_targets,
            iou_threshold=iou_thresh,
            num_classes=config.NUM_CLASSES
        )
        
        results[f'mAP@{iou_thresh}'] = map_metrics['mAP']
        
        # 클래스별 AP
        for key, value in map_metrics.items():
            if key.startswith('AP_class'):
                results[f'{key}@{iou_thresh}'] = value
    
    # Confusion Matrix 메트릭
    cm_metrics = confusion_matrix.get_metrics()
    results.update(cm_metrics)
    
    return results


def print_results(results, config):
    """결과 출력"""
    print("\n" + "=" * 60)
    print("평가 결과")
    print("=" * 60)
    
    # mAP 결과
    print("\n[mAP]")
    for key, value in results.items():
        if key.startswith('mAP'):
            print(f"  {key}: {value:.4f}")
    
    # 클래스별 AP
    print("\n[클래스별 AP]")
    for class_id in range(1, config.NUM_CLASSES):
        class_name = config.get_class_name(class_id)
        print(f"\n  {class_name}:")
        
        for key, value in results.items():
            if f'AP_class_{class_id}' in key:
                iou_threshold = key.split('@')[-1]
                print(f"    AP@{iou_threshold}: {value:.4f}")
    
    # 기타 메트릭
    print("\n[분류 메트릭]")
    for key in ['precision', 'recall', 'f1_score']:
        if key in results:
            print(f"  {key}: {results[key]:.4f}")
    
    print("=" * 60)


def save_results(results, output_path):
    """결과 JSON으로 저장"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ 결과 저장: {output_path}")


def main(args):
    """메인 함수"""
    
    # Config 로드
    config = Config()
    
    # 디바이스 설정
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"사용 디바이스: {device}")
    
    # 데이터 로더 생성 (테스트 데이터만)
    print("\n데이터 로더 생성 중...")
    _, _, test_loader = get_data_loaders(
        data_root=args.data_root,
        config=config
    )
    
    # 모델 생성
    print("\n모델 생성 중...")
    model = create_model(
        num_classes=config.NUM_CLASSES,
        pretrained=False
    )
    model.to(device)
    
    # 체크포인트 로드
    print(f"\n체크포인트 로드 중: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print("✓ 모델 로드 완료")
    
    # 평가
    results = evaluate_model(
        model,
        test_loader,
        device,
        config,
        iou_thresholds=args.iou_thresholds
    )
    
    # 결과 출력
    print_results(results, config)
    
    # 결과 저장
    if args.output:
        save_results(results, args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mask R-CNN 평가")
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="체크포인트 경로"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="데이터 루트 디렉토리"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation_results.json",
        help="결과 저장 경로"
    )
    parser.add_argument(
        "--iou_thresholds",
        type=float,
        nargs='+',
        default=[0.5, 0.75, 0.9],
        help="IoU 임계값 리스트"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="디바이스 (cuda 또는 cpu)"
    )
    
    args = parser.parse_args()
    main(args)
