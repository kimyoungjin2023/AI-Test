"""
체크포인트 저장 및 로드
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Optional, Any
import shutil


class CheckpointManager:
    """체크포인트 관리 클래스"""
    
    def __init__(self, checkpoint_dir: str, max_to_keep: int = 5):
        """
        Args:
            checkpoint_dir: 체크포인트 저장 디렉토리
            max_to_keep: 유지할 최대 체크포인트 수
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_to_keep = max_to_keep
        self.best_metric = float('-inf')
        self.checkpoints = []
    
    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: Dict[str, float],
        is_best: bool = False,
        extra_state: Optional[Dict] = None
    ) -> str:
        """
        체크포인트 저장
        
        Args:
            model: 모델
            optimizer: 옵티마이저
            epoch: 에폭
            metrics: 메트릭
            is_best: 최고 성능 체크포인트 여부
            extra_state: 추가 상태 정보
            
        Returns:
            저장된 체크포인트 경로
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics
        }
        
        if extra_state:
            checkpoint.update(extra_state)
        
        # 일반 체크포인트 저장
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)
        self.checkpoints.append(checkpoint_path)
        
        # 최고 성능 체크포인트 저장
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pth"
            shutil.copy2(checkpoint_path, best_path)
            print(f"✓ 최고 성능 모델 저장: {best_path}")
        
        # 최신 체크포인트 저장
        latest_path = self.checkpoint_dir / "latest_model.pth"
        shutil.copy2(checkpoint_path, latest_path)
        
        # 오래된 체크포인트 삭제
        self._cleanup_old_checkpoints()
        
        return str(checkpoint_path)
    
    def load_checkpoint(
        self,
        checkpoint_path: str,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: str = 'cuda'
    ) -> Dict[str, Any]:
        """
        체크포인트 로드
        
        Args:
            checkpoint_path: 체크포인트 경로
            model: 모델
            optimizer: 옵티마이저 (선택적)
            device: 디바이스
            
        Returns:
            체크포인트 딕셔너리
        """
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # 모델 가중치 로드
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 옵티마이저 상태 로드
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        print(f"✓ 체크포인트 로드: {checkpoint_path}")
        print(f"  - Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"  - Metrics: {checkpoint.get('metrics', {})}")
        
        return checkpoint
    
    def load_best_checkpoint(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: str = 'cuda'
    ) -> Dict[str, Any]:
        """최고 성능 체크포인트 로드"""
        best_path = self.checkpoint_dir / "best_model.pth"
        if not best_path.exists():
            raise FileNotFoundError(f"최고 성능 체크포인트를 찾을 수 없습니다: {best_path}")
        
        return self.load_checkpoint(str(best_path), model, optimizer, device)
    
    def load_latest_checkpoint(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: str = 'cuda'
    ) -> Dict[str, Any]:
        """최신 체크포인트 로드"""
        latest_path = self.checkpoint_dir / "latest_model.pth"
        if not latest_path.exists():
            raise FileNotFoundError(f"최신 체크포인트를 찾을 수 없습니다: {latest_path}")
        
        return self.load_checkpoint(str(latest_path), model, optimizer, device)
    
    def _cleanup_old_checkpoints(self):
        """오래된 체크포인트 삭제"""
        if len(self.checkpoints) > self.max_to_keep:
            # 오래된 것부터 삭제
            for old_checkpoint in self.checkpoints[:-self.max_to_keep]:
                if old_checkpoint.exists():
                    old_checkpoint.unlink()
                    print(f"오래된 체크포인트 삭제: {old_checkpoint.name}")
            
            self.checkpoints = self.checkpoints[-self.max_to_keep:]


class EarlyStopping:
    """조기 종료 클래스"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'max'):
        """
        Args:
            patience: 개선이 없을 때 기다릴 에폭 수
            min_delta: 개선으로 간주할 최소 변화량
            mode: 'max' 또는 'min' (메트릭 최대화/최소화)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, metric: float) -> bool:
        """
        메트릭 업데이트 및 조기 종료 판단
        
        Args:
            metric: 현재 메트릭 값
            
        Returns:
            조기 종료 여부
        """
        score = metric if self.mode == 'max' else -metric
        
        if self.best_score is None:
            self.best_score = score
            return False
        
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                print(f"\n조기 종료: {self.patience} 에폭 동안 개선 없음")
                return True
        
        return False
    
    def reset(self):
        """상태 초기화"""
        self.counter = 0
        self.best_score = None
        self.early_stop = False


if __name__ == "__main__":
    print("CheckpointManager 테스트")
    print("=" * 60)
    
    # 체크포인트 매니저 생성
    manager = CheckpointManager(checkpoint_dir="./test_checkpoints", max_to_keep=3)
    print("✓ CheckpointManager 생성 완료")
    
    # EarlyStopping 테스트
    early_stopping = EarlyStopping(patience=3, mode='max')
    
    metrics_sequence = [0.5, 0.6, 0.65, 0.64, 0.63, 0.62]
    for epoch, metric in enumerate(metrics_sequence, 1):
        should_stop = early_stopping(metric)
        print(f"Epoch {epoch}: metric={metric:.2f}, counter={early_stopping.counter}")
        if should_stop:
            break
    
    print("\n✓ CheckpointManager 및 EarlyStopping 테스트 완료")
