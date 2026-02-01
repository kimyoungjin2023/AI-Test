"""
계층적 디렉토리 구조를 위한 데이터 분할 유틸리티

원본 데이터가 이미 train/val/test로 분할되어 있다면 이 스크립트는 필요없습니다.
만약 전체 데이터를 분할해야 한다면 사용하세요.
"""

import os
import random
import shutil
from pathlib import Path
from typing import Tuple
import json


def split_hierarchical_dataset(
    source_dir: str,
    output_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
):
    """
    계층적 구조의 데이터셋을 train/val/test로 분할
    
    원본 구조:
        source_dir/
        ├── images/
        │   ├── colon/
        │   │   ├── cancer/
        │   │   ├── polyp/
        │   │   └── ulcer/
        │   └── stomach/
        │       ├── cancer/
        │       ├── polyp/
        │       └── ulcer/
        └── annotations/
            └── (동일 구조)
    
    출력 구조:
        output_dir/
        ├── train/
        │   ├── images/
        │   │   └── (동일 계층 구조)
        │   └── annotations/
        │       └── (동일 계층 구조)
        ├── val/
        └── test/
    
    Args:
        source_dir: 원본 데이터 디렉토리
        output_dir: 출력 디렉토리
        train_ratio: 학습 데이터 비율
        val_ratio: 검증 데이터 비율
        test_ratio: 테스트 데이터 비율
        seed: 랜덤 시드
    """
    random.seed(seed)
    
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    # 출력 디렉토리 생성
    for split in ['train', 'val', 'test']:
        for organ in ['stomach', 'colon']:
            for lesion in ['cancer', 'polyp', 'ulcer']:
                (output_path / split / 'images' / organ / lesion).mkdir(parents=True, exist_ok=True)
                (output_path / split / 'annotations' / organ / lesion).mkdir(parents=True, exist_ok=True)
    
    # 각 organ/lesion 조합에 대해 분할
    total_files = 0
    split_counts = {'train': 0, 'val': 0, 'test': 0}
    
    for organ in ['stomach', 'colon']:
        for lesion in ['cancer', 'polyp', 'ulcer']:
            print(f"\n처리 중: {organ}/{lesion}")
            
            images_dir = source_path / 'images' / organ / lesion
            annotations_dir = source_path / 'annotations' / organ / lesion
            
            if not images_dir.exists():
                print(f"  경고: 디렉토리 없음 - {images_dir}")
                continue
            
            # 이미지 파일 리스트
            image_files = sorted(list(images_dir.glob('*.png')))
            
            if len(image_files) == 0:
                print(f"  경고: 이미지 없음 - {images_dir}")
                continue
            
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
                for img_file in files:
                    # 이미지 복사
                    dest_img = output_path / split_name / 'images' / organ / lesion / img_file.name
                    shutil.copy2(img_file, dest_img)
                    
                    # 어노테이션 복사
                    ann_file = annotations_dir / f"{img_file.stem}.json"
                    if ann_file.exists():
                        dest_ann = output_path / split_name / 'annotations' / organ / lesion / ann_file.name
                        shutil.copy2(ann_file, dest_ann)
                    else:
                        print(f"  경고: 어노테이션 없음 - {ann_file.name}")
                
                split_counts[split_name] += len(files)
            
            total_files += total
            print(f"  총 {total}개 파일 분할됨")
    
    # 통계 출력
    print("\n" + "=" * 60)
    print("데이터 분할 완료")
    print("=" * 60)
    print(f"총 데이터: {total_files}개")
    print(f"학습: {split_counts['train']}개 ({train_ratio*100:.1f}%)")
    print(f"검증: {split_counts['val']}개 ({val_ratio*100:.1f}%)")
    print(f"테스트: {split_counts['test']}개 ({test_ratio*100:.1f}%)")
    print("=" * 60)


def verify_dataset_structure(data_root: str):
    """
    데이터셋 구조 검증
    
    Args:
        data_root: 데이터 루트 디렉토리
    """
    data_path = Path(data_root)
    
    print("\n" + "=" * 60)
    print("데이터셋 구조 검증")
    print("=" * 60)
    
    for split in ['train', 'val', 'test']:
        split_path = data_path / split
        
        if not split_path.exists():
            print(f"\n{split.upper()}: 디렉토리 없음")
            continue
        
        print(f"\n{split.upper()}:")
        total_images = 0
        total_annotations = 0
        
        for organ in ['stomach', 'colon']:
            for lesion in ['cancer', 'polyp', 'ulcer']:
                images_dir = split_path / 'images' / organ / lesion
                annotations_dir = split_path / 'annotations' / organ / lesion
                
                if images_dir.exists():
                    num_images = len(list(images_dir.glob('*.png')))
                    num_annotations = len(list(annotations_dir.glob('*.json'))) if annotations_dir.exists() else 0
                    
                    total_images += num_images
                    total_annotations += num_annotations
                    
                    print(f"  {organ}/{lesion}: {num_images} images, {num_annotations} annotations")
        
        print(f"  총계: {total_images} images, {total_annotations} annotations")
        
        if total_images != total_annotations:
            print(f"  ⚠ 경고: 이미지와 어노테이션 수가 일치하지 않습니다!")
    
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="계층적 데이터셋 분할")
    parser.add_argument('--source', type=str, required=True, help="원본 데이터 디렉토리")
    parser.add_argument('--output', type=str, required=True, help="출력 디렉토리")
    parser.add_argument('--train_ratio', type=float, default=0.8, help="학습 데이터 비율")
    parser.add_argument('--val_ratio', type=float, default=0.1, help="검증 데이터 비율")
    parser.add_argument('--test_ratio', type=float, default=0.1, help="테스트 데이터 비율")
    parser.add_argument('--seed', type=int, default=42, help="랜덤 시드")
    parser.add_argument('--verify_only', action='store_true', help="검증만 수행")
    
    args = parser.parse_args()
    
    if args.verify_only:
        verify_dataset_structure(args.output)
    else:
        split_hierarchical_dataset(
            args.source,
            args.output,
            args.train_ratio,
            args.val_ratio,
            args.test_ratio,
            args.seed
        )
        verify_dataset_structure(args.output)
