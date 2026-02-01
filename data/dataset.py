"""
AI Hub 내시경 이미지 데이터셋 로더 (계층적 디렉토리 구조 버전)
JSON 어노테이션을 파싱하고 Mask R-CNN 입력 형식으로 변환합니다.

수정 사항:
- 계층적 디렉토리 구조 지원 (organ/lesion/)
- 디렉토리 경로에서 클래스 정보 추출
- 재귀적 파일 검색
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

from config.config import Config


class EndoscopyDataset(Dataset):
    """
    AI Hub 내시경 이미지 데이터셋 클래스 (계층적 구조 버전)
    
    데이터 구조:
        data_root/
        ├── train/
        │   ├── images/
        │   │   ├── colon/
        │   │   │   ├── cancer/
        │   │   │   │   ├── 1_1_00001.png
        │   │   │   │   └── ...
        │   │   │   ├── polyp/
        │   │   │   └── ulcer/
        │   │   └── stomach/
        │   │       ├── cancer/
        │   │       ├── polyp/
        │   │       └── ulcer/
        │   └── annotations/
        │       └── (동일 구조)
    """
    
    # 클래스 매핑 (디렉토리 이름 → 클래스 ID)
    ORGAN_MAP = {
        'stomach': 0,
        'colon': 1
    }
    
    LESION_MAP = {
        'ulcer': 0,
        'polyp': 1,
        'cancer': 2
    }
    
    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        transforms: Optional[Any] = None,
        config: Optional[Config] = None
    ):
        """
        Args:
            data_root: 데이터 루트 디렉토리
            split: 'train', 'val', 'test' 중 하나
            transforms: albumentations 변환 객체
            config: Config 객체
        """
        self.data_root = Path(data_root)
        self.split = split
        self.transforms = transforms
        self.config = config or Config()
        
        self.split_dir = self.data_root / split
        self.images_root = self.split_dir / "images"
        self.annotations_root = self.split_dir / "annotations"
        
        # 이미지 파일 및 메타데이터 수집
        self.samples = self._collect_samples()
        
        if len(self.samples) == 0:
            raise ValueError(f"이미지를 찾을 수 없습니다: {self.images_root}")
        
        print(f"✓ {split.upper()} 데이터셋: {len(self.samples)}개 샘플 로드됨")
        self._print_class_distribution()
    
    def _collect_samples(self) -> List[Dict[str, Any]]:
        """
        계층적 디렉토리 구조에서 모든 샘플 수집
        
        Returns:
            샘플 정보 리스트 [{'image_path': ..., 'annotation_path': ..., 'organ': ..., 'lesion': ...}]
        """
        samples = []
        
        # 모든 조합 탐색
        for organ in ['stomach', 'colon']:
            for lesion in ['cancer', 'polyp', 'ulcer']:
                images_dir = self.images_root / organ / lesion
                annotations_dir = self.annotations_root / organ / lesion
                
                # 해당 디렉토리가 존재하는지 확인
                if not images_dir.exists():
                    print(f"경고: 디렉토리 없음 - {images_dir}")
                    continue
                
                # PNG 파일 수집
                image_files = sorted(list(images_dir.glob("*.png")))
                
                for img_path in image_files:
                    # 대응하는 어노테이션 찾기
                    ann_path = annotations_dir / f"{img_path.stem}.json"
                    
                    if not ann_path.exists():
                        print(f"경고: 어노테이션 없음 - {ann_path}")
                        continue
                    
                    samples.append({
                        'image_path': img_path,
                        'annotation_path': ann_path,
                        'organ': organ,
                        'lesion': lesion,
                        'organ_id': self.ORGAN_MAP[organ],
                        'lesion_id': self.LESION_MAP[lesion]
                    })
        
        return samples
    
    def _print_class_distribution(self):
        """클래스 분포 출력"""
        from collections import Counter
        
        class_counts = Counter()
        for sample in self.samples:
            organ = sample['organ']
            lesion = sample['lesion']
            class_counts[f"{organ}_{lesion}"] += 1
        
        print(f"  클래스 분포:")
        for class_name, count in sorted(class_counts.items()):
            print(f"    {class_name}: {count}개")
    
    def __len__(self) -> int:
        """데이터셋 크기 반환"""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        데이터셋에서 하나의 샘플 반환
        
        Returns:
            image: (C, H, W) 텐서
            target: dict with keys:
                - boxes: (N, 4) 바운딩 박스 [x1, y1, x2, y2]
                - labels: (N,) 클래스 레이블
                - masks: (N, H, W) 세그멘테이션 마스크
                - image_id: 이미지 ID
                - area: (N,) 객체 면적
                - iscrowd: (N,) crowd 여부
        """
        sample = self.samples[idx]
        
        # 이미지 로드
        image = cv2.imread(str(sample['image_path']))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 어노테이션 로드
        with open(sample['annotation_path'], 'r', encoding='utf-8') as f:
            annotation = json.load(f)
        
        # 타겟 파싱 (디렉토리 기반 클래스 정보 사용)
        boxes, labels, masks = self._parse_annotation(
            annotation, 
            image.shape[:2],
            sample['organ_id'],
            sample['lesion_id']
        )
        
        # 데이터 증강 적용
        if self.transforms is not None and len(boxes) > 0:
            # masks를 numpy array로 변환 (albumentations 호환)
            masks_array = np.array(masks) if len(masks) > 0 else np.zeros((0, image.shape[0], image.shape[1]), dtype=np.uint8)
            
            try:
                transformed = self.transforms(
                    image=image,
                    masks=masks_array,
                    bboxes=boxes,
                    labels=labels
                )
                image = transformed['image']
                masks = list(transformed['masks']) if len(transformed['masks']) > 0 else []
                boxes = list(transformed['bboxes'])
                labels = list(transformed['labels'])
            except Exception as e:
                print(f"변환 오류 (idx={idx}): {e}")
                # 변환 실패 시 원본 사용
        
        # NumPy to Tensor 변환
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        # 정규화
        mean = torch.tensor(self.config.IMAGE_MEAN).view(3, 1, 1)
        std = torch.tensor(self.config.IMAGE_STD).view(3, 1, 1)
        image = (image - mean) / std
        
        # 타겟 딕셔너리 생성
        target = self._create_target(boxes, labels, masks, idx)
        
        return image, target
    
    def _parse_annotation(
        self,
        annotation: Dict,
        image_shape: Tuple[int, int],
        organ_id: int,
        lesion_id: int
    ) -> Tuple[List, List, List]:
        """
        JSON 어노테이션 파싱
        
        Args:
            annotation: JSON 어노테이션 딕셔너리
            image_shape: (height, width)
            organ_id: 장기 ID (디렉토리에서 추출)
            lesion_id: 병변 ID (디렉토리에서 추출)
            
        Returns:
            boxes: List of [x1, y1, x2, y2]
            labels: List of class IDs
            masks: List of binary masks
        """
        height, width = image_shape
        shapes = annotation.get('shapes', [])
        
        boxes = []
        labels = []
        masks = []
        
        # 클래스 ID 계산 (디렉토리 기반)
        class_id = self.config.get_class_id(organ_id, lesion_id)
        
        for shape in shapes:
            points = np.array(shape['points'], dtype=np.float32)
            shape_type = shape.get('shape_type', 'polygon')
            
            # 좌표 유효성 검사
            if len(points) < 3:  # 최소 3개 점 필요
                print(f"경고: 점이 부족합니다 (shape_type={shape_type}, points={len(points)})")
                continue
            
            if shape_type == 'rectangle':
                # 직사각형: 2개 점
                if len(points) < 2:
                    continue
                    
                x1, y1 = points[0]
                x2, y2 = points[1]
                
                # 좌표 정렬 (x1 < x2, y1 < y2)
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)
                
                # 경계 체크
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width, x2), min(height, y2)
                
                # 최소 크기 체크
                if x2 - x1 < 1 or y2 - y1 < 1:
                    continue
                
                # 바운딩 박스
                boxes.append([x1, y1, x2, y2])
                
                # 마스크 생성 (직사각형 영역)
                mask = np.zeros((height, width), dtype=np.uint8)
                cv2.rectangle(
                    mask,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    1,
                    -1
                )
                masks.append(mask)
                
            elif shape_type == 'polygon':
                # 다각형
                points_int = points.astype(np.int32)
                
                # 바운딩 박스 계산
                x1, y1 = points.min(axis=0)
                x2, y2 = points.max(axis=0)
                
                # 경계 체크
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width, x2), min(height, y2)
                
                # 최소 크기 체크
                if x2 - x1 < 1 or y2 - y1 < 1:
                    continue
                
                boxes.append([x1, y1, x2, y2])
                
                # 마스크 생성 (다각형 영역)
                mask = np.zeros((height, width), dtype=np.uint8)
                cv2.fillPoly(mask, [points_int], 1)
                masks.append(mask)
            
            else:
                print(f"경고: 지원하지 않는 shape_type: {shape_type}")
                continue
            
            labels.append(class_id)
        
        return boxes, labels, masks
    
    def _create_target(
        self,
        boxes: List,
        labels: List,
        masks: List,
        image_id: int
    ) -> Dict[str, torch.Tensor]:
        """
        타겟 딕셔너리 생성
        
        Args:
            boxes: List of bounding boxes
            labels: List of class labels
            masks: List of segmentation masks
            image_id: Image ID
            
        Returns:
            target: Dictionary of tensors
        """
        num_objs = len(boxes)
        
        if num_objs == 0:
            # 객체가 없는 경우 빈 텐서 반환
            return {
                'boxes': torch.zeros((0, 4), dtype=torch.float32),
                'labels': torch.zeros((0,), dtype=torch.int64),
                'masks': torch.zeros((0, 1, 1), dtype=torch.uint8),
                'image_id': torch.tensor([image_id]),
                'area': torch.zeros((0,), dtype=torch.float32),
                'iscrowd': torch.zeros((0,), dtype=torch.int64)
            }
        
        # 박스를 텐서로 변환
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        # 마스크를 텐서로 변환 (N, H, W)
        masks = torch.as_tensor(np.array(masks), dtype=torch.uint8)
        
        # 면적 계산
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        
        # iscrowd (모든 객체가 개별 인스턴스)
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'masks': masks,
            'image_id': torch.tensor([image_id]),
            'area': area,
            'iscrowd': iscrowd
        }
        
        return target
    
    def get_image_info(self, idx: int) -> Dict[str, Any]:
        """이미지 정보 반환 (디버깅용)"""
        sample = self.samples[idx]
        
        with open(sample['annotation_path'], 'r', encoding='utf-8') as f:
            annotation = json.load(f)
        
        return {
            'image_path': str(sample['image_path']),
            'annotation_path': str(sample['annotation_path']),
            'organ': sample['organ'],
            'lesion': sample['lesion'],
            'num_objects': len(annotation.get('shapes', [])),
            'image_size': (annotation.get('imageHeight'), annotation.get('imageWidth'))
        }


def collate_fn(batch):
    """
    배치 데이터 collate 함수
    
    각 이미지는 객체 수가 다르므로 리스트로 반환
    """
    images = []
    targets = []
    
    for image, target in batch:
        images.append(image)
        targets.append(target)
    
    # 이미지는 스택, 타겟은 리스트로 유지
    images = torch.stack(images, dim=0)
    
    return images, targets


if __name__ == "__main__":
    # 테스트 코드
    print("EndoscopyDataset 테스트 (계층적 구조)")
    print("=" * 60)
    
    # 테스트용 데이터셋 생성
    test_data_root = "datasets"  # 실제 경로로 변경
    
    if not os.path.exists(test_data_root):
        print(f"경고: 테스트 데이터 경로가 존재하지 않습니다: {test_data_root}")
        print("실제 데이터 경로로 변경하여 테스트하세요.")
    else:
        try:
            dataset = EndoscopyDataset(
                data_root=test_data_root,
                split='train'
            )
            
            print(f"\n데이터셋 크기: {len(dataset)}")
            
            # 첫 번째 샘플 확인
            if len(dataset) > 0:
                image, target = dataset[0]
                print(f"\n이미지 shape: {image.shape}")
                print(f"박스 개수: {len(target['boxes'])}")
                print(f"라벨: {target['labels']}")
                
                # 이미지 정보 출력
                info = dataset.get_image_info(0)
                print(f"\n이미지 정보:")
                for key, value in info.items():
                    print(f"  {key}: {value}")
        except Exception as e:
            print(f"오류 발생: {e}")
            import traceback
            traceback.print_exc()
