"""
프로젝트 코드 검증 스크립트
모든 모듈의 import와 기본 기능을 테스트합니다.
"""

import sys
from pathlib import Path

print("=" * 60)
print("Mask R-CNN 프로젝트 검증 스크립트")
print("=" * 60)

# 1. 기본 Python 모듈 체크
print("\n[1] 기본 모듈 체크...")
try:
    import json
    import os
    from pathlib import Path
    from typing import Dict, List, Tuple, Optional
    print("✓ 기본 Python 모듈 정상")
except Exception as e:
    print(f"✗ 오류: {e}")
    sys.exit(1)

# 2. Config 모듈 체크
print("\n[2] Config 모듈 체크...")
try:
    from config.config import Config, validate_config
    from config.model_config import get_default_configs
    
    config = Config()
    validate_config()
    configs = get_default_configs()
    
    print(f"✓ Config 로드 성공")
    print(f"  - 클래스 수: {config.NUM_CLASSES}")
    print(f"  - 배치 크기: {config.BATCH_SIZE}")
    print(f"  - 학습 에폭: {config.EPOCHS}")
except Exception as e:
    print(f"✗ Config 오류: {e}")
    import traceback
    traceback.print_exc()

# 3. 필수 외부 라이브러리 체크
print("\n[3] 필수 라이브러리 체크...")
required_packages = {
    'torch': 'PyTorch',
    'torchvision': 'TorchVision',
    'cv2': 'OpenCV',
    'numpy': 'NumPy',
    'PIL': 'Pillow',
    'albumentations': 'Albumentations',
    'tqdm': 'tqdm',
    'matplotlib': 'Matplotlib'
}

missing_packages = []
for package, name in required_packages.items():
    try:
        __import__(package)
        print(f"✓ {name} 설치됨")
    except ImportError:
        print(f"✗ {name} 미설치")
        missing_packages.append(name)

if missing_packages:
    print(f"\n경고: 다음 패키지를 설치해야 합니다:")
    print(f"pip install {' '.join(missing_packages).lower()}")
    print("\n계속 진행합니다 (일부 기능 제한됨)...")

# 4. 프로젝트 구조 체크
print("\n[4] 프로젝트 구조 체크...")
required_files = [
    'config/config.py',
    'config/model_config.py',
    'data/dataset.py',
    'data/loader.py',
    'data/transforms.py',
    'models/maskrcnn.py',
    'utils/logger.py',
    'utils/metrics.py',
    'utils/visualize.py',
    'utils/checkpoint.py',
    'train.py',
    'inference.py',
    'evaluate.py',
    'requirements.txt',
    'README.md'
]

project_root = Path(__file__).parent
all_files_exist = True

for file_path in required_files:
    full_path = project_root / file_path
    if full_path.exists():
        print(f"✓ {file_path}")
    else:
        print(f"✗ {file_path} 없음")
        all_files_exist = False

if all_files_exist:
    print("\n✓ 모든 필수 파일 존재")
else:
    print("\n✗ 일부 파일이 누락되었습니다")

# 5. 모듈 문법 체크 (PyTorch 없이도 가능)
print("\n[5] 모듈 문법 체크...")
import py_compile

python_files = [
    'config/config.py',
    'config/model_config.py',
    'data/dataset.py',
    'data/loader.py',
    'data/transforms.py',
    'models/maskrcnn.py',
    'utils/logger.py',
    'utils/metrics.py',
    'utils/visualize.py',
    'utils/checkpoint.py',
    'train.py',
    'inference.py',
    'evaluate.py'
]

syntax_ok = True
for py_file in python_files:
    try:
        py_compile.compile(str(project_root / py_file), doraise=True)
        print(f"✓ {py_file} 문법 정상")
    except py_compile.PyCompileError as e:
        print(f"✗ {py_file} 문법 오류: {e}")
        syntax_ok = False

# 6. 데이터 디렉토리 체크
print("\n[6] 데이터 디렉토리 체크...")
data_dir = project_root / 'data'
if data_dir.exists():
    print(f"✓ 데이터 디렉토리 존재: {data_dir}")
    
    for split in ['train', 'val', 'test']:
        split_dir = data_dir / split
        if split_dir.exists():
            images_dir = split_dir / 'images'
            annotations_dir = split_dir / 'annotations'
            
            if images_dir.exists() and annotations_dir.exists():
                num_images = len(list(images_dir.glob('*.png')))
                num_annotations = len(list(annotations_dir.glob('*.json')))
                print(f"  ✓ {split}: {num_images} images, {num_annotations} annotations")
            else:
                print(f"  ✗ {split}: images/ 또는 annotations/ 디렉토리 없음")
        else:
            print(f"  - {split}: 디렉토리 없음 (데이터 준비 필요)")
else:
    print(f"- 데이터 디렉토리 없음: {data_dir}")
    print("  데이터를 준비하고 data/ 디렉토리를 생성하세요.")

# 7. 요약
print("\n" + "=" * 60)
print("검증 요약")
print("=" * 60)

if syntax_ok:
    print("✓ 모든 Python 파일 문법 정상")
else:
    print("✗ 일부 파일에 문법 오류가 있습니다")

if missing_packages:
    print(f"⚠ {len(missing_packages)}개 패키지 미설치")
    print(f"  설치 명령: pip install -r requirements.txt")
else:
    print("✓ 모든 필수 패키지 설치됨")

if all_files_exist:
    print("✓ 프로젝트 구조 완전")
else:
    print("✗ 일부 파일 누락")

print("\n다음 단계:")
print("1. pip install -r requirements.txt")
print("2. 데이터를 data/ 디렉토리에 준비")
print("3. python train.py --data_root data --experiment_name exp_001")

print("=" * 60)
