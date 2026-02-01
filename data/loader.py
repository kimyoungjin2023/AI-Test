"""
데이터 로더 및 데이터 분할 유틸리티
"""

import os
import random
from pathlib import Path
from typing import Tuple, Optional, List
import shutil

import torch
from torch.utils.data import DataLoader, Subset
import numpy as np

from data.dataset import EndoscopyDataset, collate_fn
from data.transforms import get_train_transforms, get_val_transforms, get_test_transforms
from config.config import Config
from config.model_config import DataLoaderConfig


def set_seed(seed: int = 42):
    """재현성을 위한 시드 설정"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def split_dataset_files(
    source_dir: str,
    output_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
):
    """
    전체 데이터셋을 train/val/test로 분할
    
    Args:
        source_dir: 원본 데이터 디렉토리 (images/, annotations/ 포함)
        output_dir: 출력 디렉토리
        train_ratio: 학습 데이터 비율
        val_ratio: 검증 데이터 비율
        test_ratio: 테스트 데이터 비율
        seed: 랜덤 시드
    """
    set_seed(seed)
    
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    # 출력 디렉토리 생성
    for split in ['train', 'val', 'test']:
        (output_path / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_path / split / 'annotations').mkdir(parents=True, exist_ok=True)
    
    # 이미지 파일 리스트 수집
    images_dir = source_path / 'images'
    image_files = sorted(list(images_dir.glob('*.png')))
    
    if len(image_files) == 0:
        raise ValueError(f"이미지를 찾을 수 없습니다: {images_dir}")
    
    # 셔플
    random.shuffle(image_files)
    
    # 분할 인덱스 계산
    total = len(image_files)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    splits = {
        'train': image_files[:train_end],
        'val': image_files[train_end:val_end],
        'test': image_files[val_end:]
    }
    
    # 파일 복사
    for split_name, files in splits.items():
        print(f"\n{split_name.upper()} 분할 처리 중... ({len(files)}개 파일)")
        
        for img_file in files:
            # 이미지 복사
            dest_img = output_path / split_name / 'images' / img_file.name
            shutil.copy2(img_file, dest_img)
            
            # 어노테이션 복사
            ann_file = source_path / 'annotations' / f"{img_file.stem}.json"
            if ann_file.exists():
                dest_ann = output_path / split_name / 'annotations' / ann_file.name
                shutil.copy2(ann_file, dest_ann)
            else:
                print(f"경고: 어노테이션 파일 없음 - {ann_file.name}")
    
    # 통계 출력
    print("\n" + "=" * 60)
    print("데이터 분할 완료")
    print("=" * 60)
    print(f"총 데이터: {total}개")
    print(f"학습: {len(splits['train'])}개 ({train_ratio*100:.1f}%)")
    print(f"검증: {len(splits['val'])}개 ({val_ratio*100:.1f}%)")
    print(f"테스트: {len(splits['test'])}개 ({test_ratio*100:.1f}%)")
    print("=" * 60)


def get_data_loaders(
    data_root: str,
    config: Optional[Config] = None,
    loader_config: Optional[DataLoaderConfig] = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    데이터 로더 생성
    
    Args:
        data_root: 데이터 루트 디렉토리 (train/val/test 포함)
        config: 전역 설정
        loader_config: 데이터로더 설정
        
    Returns:
        train_loader, val_loader, test_loader
    """
    config = config or Config()
    loader_config = loader_config or DataLoaderConfig()
    
    # 변환 생성
    train_transform = get_train_transforms(config)
    val_transform = get_val_transforms(config)
    test_transform = get_test_transforms(config)
    
    # 데이터셋 생성
    train_dataset = EndoscopyDataset(
        data_root=data_root,
        split='train',
        transforms=train_transform,
        config=config
    )
    
    val_dataset = EndoscopyDataset(
        data_root=data_root,
        split='val',
        transforms=val_transform,
        config=config
    )
    
    test_dataset = EndoscopyDataset(
        data_root=data_root,
        split='test',
        transforms=test_transform,
        config=config
    )
    
    # 데이터 로더 생성
    train_loader = DataLoader(
        train_dataset,
        batch_size=loader_config.batch_size,
        shuffle=loader_config.shuffle_train,
        num_workers=loader_config.num_workers,
        collate_fn=collate_fn,
        pin_memory=loader_config.pin_memory,
        drop_last=loader_config.drop_last
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=loader_config.batch_size,
        shuffle=loader_config.shuffle_val,
        num_workers=loader_config.num_workers,
        collate_fn=collate_fn,
        pin_memory=loader_config.pin_memory,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # 테스트는 배치 크기 1
        shuffle=False,
        num_workers=loader_config.num_workers,
        collate_fn=collate_fn,
        pin_memory=loader_config.pin_memory,
        drop_last=False
    )
    
    print("\n" + "=" * 60)
    print("데이터 로더 생성 완료")
    print("=" * 60)
    print(f"학습 배치 수: {len(train_loader)}")
    print(f"검증 배치 수: {len(val_loader)}")
    print(f"테스트 배치 수: {len(test_loader)}")
    print(f"배치 크기: {loader_config.batch_size}")
    print(f"워커 수: {loader_config.num_workers}")
    print("=" * 60)
    
    return train_loader, val_loader, test_loader


class DataLoaderBuilder:
    """데이터 로더 빌더 클래스 (고급 사용)"""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.train_transform = None
        self.val_transform = None
        self.test_transform = None
        self.batch_size = 4
        self.num_workers = 4
    
    def set_transforms(self, train_transform, val_transform, test_transform):
        """커스텀 변환 설정"""
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform
        return self
    
    def set_batch_size(self, batch_size: int):
        """배치 크기 설정"""
        self.batch_size = batch_size
        return self
    
    def set_num_workers(self, num_workers: int):
        """워커 수 설정"""
        self.num_workers = num_workers
        return self
    
    def build(self, data_root: str) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """데이터 로더 빌드"""
        # 기본 변환 설정
        if self.train_transform is None:
            self.train_transform = get_train_transforms(self.config)
        if self.val_transform is None:
            self.val_transform = get_val_transforms(self.config)
        if self.test_transform is None:
            self.test_transform = get_test_transforms(self.config)
        
        # 데이터셋 생성
        train_dataset = EndoscopyDataset(
            data_root=data_root,
            split='train',
            transforms=self.train_transform,
            config=self.config
        )
        
        val_dataset = EndoscopyDataset(
            data_root=data_root,
            split='val',
            transforms=self.val_transform,
            config=self.config
        )
        
        test_dataset = EndoscopyDataset(
            data_root=data_root,
            split='test',
            transforms=self.test_transform,
            config=self.config
        )
        
        # 데이터 로더 생성
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True
        )
        
        return train_loader, val_loader, test_loader


def analyze_dataset(data_root: str):
    """
    데이터셋 분석 및 통계 출력
    
    Args:
        data_root: 데이터 루트 디렉토리
    """
    print("\n" + "=" * 60)
    print("데이터셋 분석")
    print("=" * 60)
    
    for split in ['train', 'val', 'test']:
        try:
            dataset = EndoscopyDataset(
                data_root=data_root,
                split=split,
                transforms=None
            )
            
            print(f"\n{split.upper()} 데이터셋:")
            print(f"  총 이미지 수: {len(dataset)}")
            
            # 클래스 분포 분석
            class_counts = {i: 0 for i in range(Config.NUM_CLASSES)}
            total_objects = 0
            
            for idx in range(len(dataset)):
                _, target = dataset[idx]
                labels = target['labels'].numpy()
                
                for label in labels:
                    class_counts[label] += 1
                    total_objects += 1
            
            print(f"  총 객체 수: {total_objects}")
            print(f"  클래스 분포:")
            for class_id, count in class_counts.items():
                if count > 0:
                    class_name = Config.get_class_name(class_id)
                    percentage = (count / total_objects) * 100
                    print(f"    {class_name}: {count} ({percentage:.2f}%)")
        
        except Exception as e:
            print(f"\n{split.upper()} 데이터셋 로드 실패: {e}")
    
    print("=" * 60)


if __name__ == "__main__":
    # 테스트 코드
    print("DataLoader 테스트")
    print("=" * 60)
    
    # 시드 설정 테스트
    set_seed(42)
    print("✓ 시드 설정 완료")
    
    # 데이터 로더 빌더 테스트
    builder = DataLoaderBuilder()
    builder.set_batch_size(2).set_num_workers(2)
    print("✓ 데이터 로더 빌더 생성 완료")
    
    print("\n참고: 실제 데이터로 테스트하려면 data_root 경로를 설정하세요.")
