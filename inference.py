"""
Mask R-CNN 추론 스크립트
단일 이미지 또는 디렉토리에 대한 추론 수행
"""

import argparse
import torch
import cv2
import numpy as np
from pathlib import Path
import sys
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent))

from config.config import Config
from models.maskrcnn import create_model
from utils.checkpoint import CheckpointManager
from utils.visualize import Visualizer, save_prediction_image
from data.transforms import get_test_transforms


class Inferencer:
    """추론 클래스"""
    
    def __init__(
        self,
        checkpoint_path: str,
        config: Config = None,
        device: str = 'cuda',
        score_threshold: float = 0.5
    ):
        """
        Args:
            checkpoint_path: 체크포인트 경로
            config: Config 객체
            device: 디바이스
            score_threshold: 검출 점수 임계값
        """
        self.config = config or Config()
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.score_threshold = score_threshold
        
        # 모델 로드
        self.model = create_model(
            num_classes=self.config.NUM_CLASSES,
            pretrained=False
        )
        self.model.to(self.device)
        
        # 체크포인트 로드
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"✓ 모델 로드 완료: {checkpoint_path}")
        print(f"✓ 디바이스: {self.device}")
        
        # 변환 및 시각화
        self.transform = get_test_transforms(self.config)
        self.visualizer = Visualizer(self.config)
    
    def preprocess_image(self, image_path: str):
        """이미지 전처리"""
        # 이미지 로드
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 변환 적용
        transformed = self.transform(
            image=image,
            bboxes=[],  # 더미
            labels=[]
        )
        
        # 정규화
        image_tensor = torch.from_numpy(transformed['image']).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor(self.config.IMAGE_MEAN).view(3, 1, 1)
        std = torch.tensor(self.config.IMAGE_STD).view(3, 1, 1)
        image_tensor = (image_tensor - mean) / std
        
        return image, image_tensor
    
    @torch.no_grad()
    def predict(self, image_path: str):
        """
        단일 이미지 추론
        
        Args:
            image_path: 이미지 경로
            
        Returns:
            prediction: 예측 결과
            original_image: 원본 이미지
        """
        # 전처리
        original_image, image_tensor = self.preprocess_image(image_path)
        
        # 추론
        image_tensor = image_tensor.to(self.device)
        predictions = self.model.predict(
            [image_tensor],
            score_threshold=self.score_threshold
        )
        
        prediction = predictions[0]
        
        # 결과 출력
        num_detections = len(prediction['boxes'])
        print(f"\n검출된 객체: {num_detections}개")
        
        if num_detections > 0:
            for i in range(num_detections):
                label = prediction['labels'][i].item()
                score = prediction['scores'][i].item()
                class_name = self.config.get_class_name(label)
                print(f"  - {class_name}: {score:.3f}")
        
        return prediction, original_image
    
    def predict_and_save(self, image_path: str, output_path: str):
        """추론 및 결과 저장"""
        prediction, original_image = self.predict(image_path)
        
        # 시각화
        vis_image = self.visualizer.draw_boxes_and_masks(
            original_image,
            prediction['boxes'],
            prediction['labels'],
            prediction['scores'],
            prediction.get('masks')
        )
        
        # 저장
        vis_image_bgr = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, vis_image_bgr)
        
        print(f"✓ 결과 저장: {output_path}")
    
    def predict_directory(self, input_dir: str, output_dir: str):
        """디렉토리 내 모든 이미지 추론"""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 이미지 파일 수집
        image_files = list(input_path.glob('*.png')) + \
                      list(input_path.glob('*.jpg')) + \
                      list(input_path.glob('*.jpeg'))
        
        if len(image_files) == 0:
            print(f"이미지를 찾을 수 없습니다: {input_dir}")
            return
        
        print(f"\n총 {len(image_files)}개 이미지 처리 중...")
        
        for img_file in tqdm(image_files):
            try:
                output_file = output_path / f"pred_{img_file.name}"
                self.predict_and_save(str(img_file), str(output_file))
            except Exception as e:
                print(f"오류 발생 ({img_file.name}): {e}")
        
        print(f"\n✓ 모든 결과 저장 완료: {output_dir}")


def main(args):
    """메인 함수"""
    
    # Config 로드
    config = Config()
    
    # Inferencer 생성
    inferencer = Inferencer(
        checkpoint_path=args.checkpoint,
        config=config,
        device=args.device,
        score_threshold=args.score_threshold
    )
    
    # 단일 이미지 또는 디렉토리 추론
    if args.image:
        # 단일 이미지
        output_path = args.output if args.output else "prediction.png"
        inferencer.predict_and_save(args.image, output_path)
    
    elif args.input_dir:
        # 디렉토리
        output_dir = args.output if args.output else "predictions"
        inferencer.predict_directory(args.input_dir, output_dir)
    
    else:
        print("--image 또는 --input_dir 중 하나를 지정해야 합니다.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mask R-CNN 추론")
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="체크포인트 경로"
    )
    parser.add_argument(
        "--image",
        type=str,
        help="입력 이미지 경로 (단일 이미지)"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        help="입력 디렉토리 (여러 이미지)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="출력 경로 (파일 또는 디렉토리)"
    )
    parser.add_argument(
        "--score_threshold",
        type=float,
        default=0.5,
        help="검출 점수 임계값"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="디바이스 (cuda 또는 cpu)"
    )
    
    args = parser.parse_args()
    main(args)
