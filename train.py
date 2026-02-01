"""
Mask R-CNN 학습 스크립트
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import argparse
import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import sys
from pathlib import Path

# 프로젝트 경로 추가
sys.path.append(str(Path(__file__).parent))

from config.config import Config
from config.model_config import get_default_configs
from data.loader import get_data_loaders, set_seed
from models.maskrcnn import create_model, print_model_summary
from utils.logger import Logger, MetricLogger, ProgressLogger
from utils.checkpoint import CheckpointManager, EarlyStopping
from utils.metrics import calculate_map


def train_one_epoch(
    model,
    data_loader,
    optimizer,
    device,
    epoch,
    logger,
    use_amp=False
):
    """한 에폭 학습"""
    model.train()
    metric_logger = MetricLogger()
    scaler = GradScaler() if use_amp else None
    
    pbar = tqdm(data_loader, desc=f"Epoch {epoch} [Train]")
    
    for batch_idx, (images, targets) in enumerate(pbar):
        # 데이터를 디바이스로 이동
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Forward pass
        if use_amp:
            with autocast():
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
        else:
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
        
        # Backward pass
        optimizer.zero_grad()
        
        if use_amp:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()
        
        # 메트릭 업데이트
        metric_logger.update(
            total_loss=losses.item(),
            **{k: v.item() for k, v in loss_dict.items()}
        )
        
        # Progress bar 업데이트
        pbar.set_postfix({'loss': losses.item()})
    
    # 평균 메트릭
    avg_metrics = metric_logger.get_all_averages()
    
    # 로깅
    if logger:
        logger.log_metrics(avg_metrics, epoch, prefix="train/")
    
    return avg_metrics


@torch.no_grad()
def evaluate(model, data_loader, device, epoch, logger, config):
    """모델 평가"""
    model.eval()
    metric_logger = MetricLogger()
    
    all_predictions = []
    all_targets = []
    
    pbar = tqdm(data_loader, desc=f"Epoch {epoch} [Val]")
    
    for images, targets in pbar:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # 추론
        outputs = model(images)
        
        all_predictions.extend(outputs)
        all_targets.extend(targets)
    
    # mAP 계산
    map_metrics = calculate_map(
        all_predictions,
        all_targets,
        iou_threshold=0.5,
        num_classes=config.NUM_CLASSES
    )
    
    # 로깅
    if logger:
        logger.log_metrics(map_metrics, epoch, prefix="val/")
    
    return map_metrics


def main(args):
    """메인 학습 루프"""
    
    # 설정 로드
    config = Config()
    model_configs = get_default_configs()
    
    # 재현성
    set_seed(config.SEED)
    
    # 디바이스 설정
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"사용 디바이스: {device}")
    
    # 디렉토리 생성
    config.create_directories()
    
    # 로거 생성
    logger = Logger(
        log_dir=str(config.LOG_DIR),
        experiment_name=args.experiment_name,
        use_tensorboard=config.USE_TENSORBOARD
    )
    
    logger.info("=" * 60)
    logger.info("내시경 이미지 Mask R-CNN 학습 시작")
    logger.info("=" * 60)
    
    # 설정 저장
    logger.save_config({
        'args': vars(args),
        'config': {k: str(v) for k, v in vars(config).items() if k.isupper()}
    })
    
    # 데이터 로더 생성
    logger.info("\n데이터 로더 생성 중...")
    train_loader, val_loader, _ = get_data_loaders(
        data_root=args.data_root,
        config=config
    )
    
    # 모델 생성
    logger.info("\n모델 생성 중...")
    model = create_model(
        num_classes=config.NUM_CLASSES,
        pretrained=config.PRETRAINED,
        freeze_backbone=args.freeze_backbone
    )
    model.to(device)
    
    print_model_summary(model)
    
    # 옵티마이저
    optimizer = optim.SGD(
        model.get_trainable_parameters(),
        lr=config.LEARNING_RATE,
        momentum=config.MOMENTUM,
        weight_decay=config.WEIGHT_DECAY
    )
    
    # Learning Rate Scheduler
    if config.LR_SCHEDULER == "step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.LR_STEP_SIZE,
            gamma=config.LR_GAMMA
        )
    elif config.LR_SCHEDULER == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.EPOCHS
        )
    else:
        scheduler = None
    
    # 체크포인트 매니저
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=str(config.CHECKPOINT_DIR),
        max_to_keep=5
    )
    
    # Early Stopping
    early_stopping = EarlyStopping(
        patience=config.EARLY_STOPPING_PATIENCE,
        mode='max'
    )
    
    # 학습 시작
    progress_logger = ProgressLogger(config.EPOCHS, logger)
    progress_logger.start()
    
    best_map = 0.0
    start_epoch = 1
    
    # 체크포인트에서 재개 (선택적)
    if args.resume:
        try:
            checkpoint = checkpoint_manager.load_latest_checkpoint(
                model, optimizer, device
            )
            start_epoch = checkpoint['epoch'] + 1
            best_map = checkpoint['metrics'].get('mAP', 0.0)
            logger.info(f"체크포인트에서 재개: Epoch {start_epoch}")
        except FileNotFoundError:
            logger.warning("재개할 체크포인트를 찾을 수 없습니다. 처음부터 시작합니다.")
    
    # 학습 루프
    for epoch in range(start_epoch, config.EPOCHS + 1):
        progress_logger.epoch_start(epoch)
        
        # 학습
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, device, epoch,
            logger, use_amp=config.MIXED_PRECISION
        )
        
        # 검증
        if epoch % config.EVAL_INTERVAL == 0:
            val_metrics = evaluate(
                model, val_loader, device, epoch, logger, config
            )
            
            current_map = val_metrics['mAP']
            
            # 최고 성능 체크
            is_best = current_map > best_map
            if is_best:
                best_map = current_map
                logger.info(f"✓ 새로운 최고 mAP: {best_map:.4f}")
            
            # 체크포인트 저장
            if epoch % config.SAVE_CHECKPOINT_EVERY == 0:
                checkpoint_manager.save_checkpoint(
                    model, optimizer, epoch,
                    {'train': train_metrics, 'val': val_metrics},
                    is_best=is_best
                )
            
            # Early Stopping 체크
            if early_stopping(current_map):
                logger.info("조기 종료!")
                break
            
            progress_logger.epoch_end(epoch, {**train_metrics, **val_metrics})
        else:
            progress_logger.epoch_end(epoch, train_metrics)
        
        # Learning Rate 업데이트
        if scheduler is not None:
            scheduler.step()
    
    # 학습 종료
    progress_logger.end()
    logger.info(f"\n최고 mAP: {best_map:.4f}")
    logger.close()
    
    print("\n✓ 학습 완료!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mask R-CNN 학습")
    
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="데이터 루트 디렉토리 (train/val/test 포함)"
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="실험 이름 (기본값: 타임스탬프)"
    )
    parser.add_argument(
        "--freeze_backbone",
        action="store_true",
        help="백본 네트워크 동결"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="최신 체크포인트에서 재개"
    )
    
    args = parser.parse_args()
    
    main(args)
