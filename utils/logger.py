"""
로깅 유틸리티
학습 과정 및 평가 결과 로깅
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import json

import torch
from torch.utils.tensorboard import SummaryWriter


class Logger:
    """통합 로거 클래스"""
    
    def __init__(
        self,
        log_dir: str,
        experiment_name: Optional[str] = None,
        use_tensorboard: bool = True,
        console_level: int = logging.INFO
    ):
        """
        Args:
            log_dir: 로그 저장 디렉토리
            experiment_name: 실험 이름 (기본값: 타임스탬프)
            use_tensorboard: TensorBoard 사용 여부
            console_level: 콘솔 로그 레벨
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 실험 이름 설정
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"exp_{timestamp}"
        self.experiment_name = experiment_name
        
        # 실험 디렉토리
        self.exp_dir = self.log_dir / experiment_name
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        # 파일 로거 설정
        self.logger = self._setup_logger(console_level)
        
        # TensorBoard
        self.use_tensorboard = use_tensorboard
        self.tb_writer = None
        if use_tensorboard:
            tb_dir = self.exp_dir / "tensorboard"
            tb_dir.mkdir(exist_ok=True)
            self.tb_writer = SummaryWriter(log_dir=str(tb_dir))
            self.info("TensorBoard 활성화됨")
        
        self.info(f"로거 초기화 완료: {self.exp_dir}")
    
    def _setup_logger(self, console_level: int) -> logging.Logger:
        """파일 및 콘솔 로거 설정"""
        logger = logging.getLogger(self.experiment_name)
        logger.setLevel(logging.DEBUG)
        
        # 기존 핸들러 제거
        logger.handlers.clear()
        
        # 파일 핸들러
        log_file = self.exp_dir / f"{self.experiment_name}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        # 콘솔 핸들러
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(console_level)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        return logger
    
    def info(self, message: str):
        """INFO 레벨 로그"""
        self.logger.info(message)
    
    def debug(self, message: str):
        """DEBUG 레벨 로그"""
        self.logger.debug(message)
    
    def warning(self, message: str):
        """WARNING 레벨 로그"""
        self.logger.warning(message)
    
    def error(self, message: str):
        """ERROR 레벨 로그"""
        self.logger.error(message)
    
    def critical(self, message: str):
        """CRITICAL 레벨 로그"""
        self.logger.critical(message)
    
    def log_metrics(self, metrics: Dict[str, float], step: int, prefix: str = ""):
        """
        메트릭 로깅
        
        Args:
            metrics: 메트릭 딕셔너리
            step: 스텝 (에폭 또는 iteration)
            prefix: 메트릭 이름 접두사 (예: "train/", "val/")
        """
        # 파일 로그
        metric_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.info(f"{prefix}Step {step} - {metric_str}")
        
        # TensorBoard
        if self.tb_writer is not None:
            for key, value in metrics.items():
                tag = f"{prefix}{key}" if prefix else key
                self.tb_writer.add_scalar(tag, value, step)
    
    def log_images(
        self,
        images: Dict[str, torch.Tensor],
        step: int,
        prefix: str = ""
    ):
        """
        이미지 로깅 (TensorBoard)
        
        Args:
            images: 이미지 딕셔너리 {name: tensor}
            step: 스텝
            prefix: 태그 접두사
        """
        if self.tb_writer is not None:
            for name, img_tensor in images.items():
                tag = f"{prefix}{name}" if prefix else name
                self.tb_writer.add_image(tag, img_tensor, step)
    
    def log_text(self, text: str, tag: str, step: int):
        """텍스트 로깅 (TensorBoard)"""
        if self.tb_writer is not None:
            self.tb_writer.add_text(tag, text, step)
    
    def log_hyperparameters(self, hparams: Dict[str, Any]):
        """하이퍼파라미터 로깅"""
        # JSON 파일로 저장
        hparams_file = self.exp_dir / "hyperparameters.json"
        with open(hparams_file, 'w', encoding='utf-8') as f:
            json.dump(hparams, f, indent=2, default=str)
        
        self.info(f"하이퍼파라미터 저장: {hparams_file}")
    
    def save_config(self, config_dict: Dict[str, Any]):
        """설정 저장"""
        config_file = self.exp_dir / "config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, default=str)
        
        self.info(f"설정 저장: {config_file}")
    
    def close(self):
        """리소스 정리"""
        if self.tb_writer is not None:
            self.tb_writer.close()
            self.info("TensorBoard writer 종료")
        
        # 로거 핸들러 정리
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)


class MetricLogger:
    """
    메트릭 추적 및 평균 계산
    """
    
    def __init__(self):
        self.metrics = {}
    
    def update(self, **kwargs):
        """
        메트릭 업데이트
        
        Example:
            metric_logger.update(loss=0.5, accuracy=0.9)
        """
        for key, value in kwargs.items():
            if key not in self.metrics:
                self.metrics[key] = []
            
            # 텐서를 스칼라로 변환
            if isinstance(value, torch.Tensor):
                value = value.item()
            
            self.metrics[key].append(value)
    
    def get_average(self, key: str) -> float:
        """특정 메트릭의 평균값 반환"""
        if key not in self.metrics or len(self.metrics[key]) == 0:
            return 0.0
        return sum(self.metrics[key]) / len(self.metrics[key])
    
    def get_all_averages(self) -> Dict[str, float]:
        """모든 메트릭의 평균값 반환"""
        return {key: self.get_average(key) for key in self.metrics.keys()}
    
    def reset(self):
        """메트릭 초기화"""
        self.metrics = {}
    
    def __str__(self) -> str:
        """메트릭 문자열 표현"""
        averages = self.get_all_averages()
        return ", ".join([f"{k}: {v:.4f}" for k, v in averages.items()])


class ProgressLogger:
    """
    학습 진행 상황 로깅
    """
    
    def __init__(self, total_epochs: int, logger: Optional[Logger] = None):
        self.total_epochs = total_epochs
        self.logger = logger
        self.start_time = None
        self.epoch_start_time = None
    
    def start(self):
        """학습 시작"""
        self.start_time = datetime.now()
        if self.logger:
            self.logger.info(f"학습 시작: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            self.logger.info(f"총 에폭: {self.total_epochs}")
    
    def epoch_start(self, epoch: int):
        """에폭 시작"""
        self.epoch_start_time = datetime.now()
        if self.logger:
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Epoch {epoch}/{self.total_epochs}")
            self.logger.info(f"{'='*60}")
    
    def epoch_end(self, epoch: int, metrics: Dict[str, float]):
        """에폭 종료"""
        if self.epoch_start_time is not None:
            epoch_time = (datetime.now() - self.epoch_start_time).total_seconds()
            
            if self.logger:
                metric_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
                self.logger.info(f"Epoch {epoch} 완료 - {metric_str}")
                self.logger.info(f"에폭 소요 시간: {epoch_time:.2f}초")
    
    def end(self):
        """학습 종료"""
        if self.start_time is not None:
            total_time = (datetime.now() - self.start_time).total_seconds()
            
            if self.logger:
                self.logger.info(f"\n{'='*60}")
                self.logger.info("학습 완료!")
                self.logger.info(f"총 소요 시간: {total_time/60:.2f}분")
                self.logger.info(f"{'='*60}")


def create_logger(
    log_dir: str = "./logs",
    experiment_name: Optional[str] = None
) -> Logger:
    """
    편의 함수: Logger 생성
    """
    return Logger(
        log_dir=log_dir,
        experiment_name=experiment_name,
        use_tensorboard=True
    )


if __name__ == "__main__":
    # 테스트 코드
    print("Logger 테스트")
    print("=" * 60)
    
    # 로거 생성
    logger = create_logger(
        log_dir="./test_logs",
        experiment_name="test_experiment"
    )
    
    # 로그 메시지
    logger.info("INFO 레벨 테스트")
    logger.debug("DEBUG 레벨 테스트")
    logger.warning("WARNING 레벨 테스트")
    
    # 메트릭 로깅
    metrics = {
        'loss': 0.5,
        'accuracy': 0.9,
        'f1_score': 0.85
    }
    logger.log_metrics(metrics, step=1, prefix="train/")
    
    # 하이퍼파라미터 저장
    hparams = {
        'learning_rate': 0.001,
        'batch_size': 4,
        'epochs': 100
    }
    logger.log_hyperparameters(hparams)
    
    # MetricLogger 테스트
    print("\nMetricLogger 테스트")
    metric_logger = MetricLogger()
    metric_logger.update(loss=0.5, acc=0.9)
    metric_logger.update(loss=0.4, acc=0.92)
    print(f"평균 메트릭: {metric_logger}")
    
    # 정리
    logger.close()
    print("\n✓ Logger 테스트 완료")
