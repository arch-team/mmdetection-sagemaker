"""
Dataset utilities for downloading and preparing public datasets
"""

import os
import json
import urllib.request
import zipfile
import tarfile
import logging
from pathlib import Path
import shutil

logger = logging.getLogger(__name__)


def download_file(url, destination):
    """Download file from URL to destination"""
    logger.info(f"Downloading {url} to {destination}")
    urllib.request.urlretrieve(url, destination)


def extract_archive(archive_path, extract_to):
    """Extract archive file"""
    logger.info(f"Extracting {archive_path} to {extract_to}")
    
    if archive_path.endswith('.zip'):
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
    elif archive_path.endswith(('.tar', '.tar.gz', '.tgz')):
        with tarfile.open(archive_path, 'r') as tar_ref:
            tar_ref.extractall(extract_to)
    else:
        raise ValueError(f"Unsupported archive format: {archive_path}")


def create_coco_sample_dataset(data_dir):
    """Create a minimal COCO dataset for testing"""
    logger.info(f"Creating sample COCO dataset in {data_dir}")
    
    # Create directory structure
    annotations_dir = Path(data_dir) / "annotations"
    train_dir = Path(data_dir) / "train2017"
    val_dir = Path(data_dir) / "val2017"
    
    annotations_dir.mkdir(parents=True, exist_ok=True)
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    # Create sample images (placeholder)
    create_sample_images(train_dir, 10)
    create_sample_images(val_dir, 5)
    
    # Create COCO annotations
    create_coco_annotations(annotations_dir, train_dir, val_dir)
    
    logger.info("Sample COCO dataset created successfully")


def create_sample_images(image_dir, num_images):
    """Create sample placeholder images"""
    try:
        from PIL import Image
        import numpy as np
        
        for i in range(num_images):
            # Create a random colored image
            img_array = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            img_path = image_dir / f"sample_{i:03d}.jpg"
            img.save(img_path)
            
    except ImportError:
        logger.warning("PIL not available, creating empty image files")
        for i in range(num_images):
            img_path = image_dir / f"sample_{i:03d}.jpg"
            img_path.touch()


def create_coco_annotations(annotations_dir, train_dir, val_dir):
    """Create COCO format annotations"""
    
    # Get image files
    train_images = list(train_dir.glob("*.jpg"))
    val_images = list(val_dir.glob("*.jpg"))
    
    # Create annotations for train set
    train_annotation = create_coco_annotation_dict(train_images, "train2017")
    with open(annotations_dir / "instances_train2017.json", 'w') as f:
        json.dump(train_annotation, f)
    
    # Create annotations for val set
    val_annotation = create_coco_annotation_dict(val_images, "val2017")
    with open(annotations_dir / "instances_val2017.json", 'w') as f:
        json.dump(val_annotation, f)


def create_coco_annotation_dict(image_files, split_name):
    """Create COCO annotation dictionary"""
    
    categories = [
        {"id": 1, "name": "person", "supercategory": "person"},
        {"id": 2, "name": "car", "supercategory": "vehicle"},
        {"id": 3, "name": "dog", "supercategory": "animal"}
    ]
    
    images = []
    annotations = []
    
    for idx, img_path in enumerate(image_files):
        image_id = idx + 1
        
        # Image info
        images.append({
            "id": image_id,
            "width": 640,
            "height": 480,
            "file_name": img_path.name
        })
        
        # Create sample annotations for each image
        for ann_idx in range(2):  # 2 annotations per image
            annotation_id = image_id * 10 + ann_idx
            category_id = (ann_idx % 3) + 1
            
            # Random bounding box
            x = 50 + (ann_idx * 100)
            y = 50 + (ann_idx * 80)
            w = 100
            h = 80
            
            annotations.append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_id,
                "bbox": [x, y, w, h],
                "area": w * h,
                "iscrowd": 0
            })
    
    return {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }


def download_balloon_dataset(data_dir):
    """Download and prepare balloon dataset"""
    logger.info(f"Downloading balloon dataset to {data_dir}")
    
    balloon_url = "https://github.com/matterport/Mask_RCNN/releases/download/v2.1/balloon_dataset.zip"
    zip_path = Path(data_dir) / "balloon_dataset.zip"
    
    # Download dataset
    download_file(balloon_url, zip_path)
    
    # Extract dataset
    extract_archive(zip_path, data_dir)
    
    # Clean up zip file
    zip_path.unlink()
    
    # Convert VIA annotations to COCO format
    convert_balloon_to_coco(data_dir)
    
    logger.info("Balloon dataset downloaded and prepared")


def convert_balloon_to_coco(data_dir):
    """Convert balloon dataset VIA annotations to COCO format"""
    logger.info("Converting balloon annotations to COCO format")
    
    balloon_dir = Path(data_dir) / "balloon"
    if not balloon_dir.exists():
        logger.warning("Balloon directory not found, skipping conversion")
        return
    
    # Create COCO structure
    annotations_dir = balloon_dir / "annotations"
    annotations_dir.mkdir(exist_ok=True)
    
    # Process train and val splits
    for split in ["train", "val"]:
        split_dir = balloon_dir / split
        if not split_dir.exists():
            continue
            
        via_json = split_dir / "via_region_data.json"
        if via_json.exists():
            coco_annotation = convert_via_to_coco(via_json, split_dir)
            coco_file = annotations_dir / f"instances_{split}.json"
            with open(coco_file, 'w') as f:
                json.dump(coco_annotation, f)


def convert_via_to_coco(via_json_path, image_dir):
    """Convert VIA format to COCO format"""
    with open(via_json_path, 'r') as f:
        via_data = json.load(f)
    
    categories = [{"id": 1, "name": "balloon", "supercategory": "object"}]
    images = []
    annotations = []
    
    annotation_id = 1
    
    for image_id, (filename, file_data) in enumerate(via_data.items(), 1):
        if isinstance(file_data, dict) and 'filename' in file_data:
            filename = file_data['filename']
            
            # Image info
            images.append({
                "id": image_id,
                "width": 1024,  # Default size, should be read from actual image
                "height": 768,
                "file_name": filename
            })
            
            # Process regions (annotations)
            regions = file_data.get('regions', {})
            for region_data in regions.values():
                if 'shape_attributes' in region_data:
                    shape_attrs = region_data['shape_attributes']
                    
                    if shape_attrs.get('name') == 'polygon':
                        # Convert polygon to bounding box
                        x_coords = shape_attrs['all_points_x']
                        y_coords = shape_attrs['all_points_y']
                        
                        x_min, x_max = min(x_coords), max(x_coords)
                        y_min, y_max = min(y_coords), max(y_coords)
                        
                        bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
                        area = (x_max - x_min) * (y_max - y_min)
                        
                        annotations.append({
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": 1,
                            "bbox": bbox,
                            "area": area,
                            "iscrowd": 0
                        })
                        
                        annotation_id += 1
    
    return {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }


def download_voc2007_dataset(data_dir):
    """Download Pascal VOC 2007 dataset"""
    logger.info(f"Downloading Pascal VOC 2007 dataset to {data_dir}")
    
    voc_urls = [
        "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar",
        "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar"
    ]
    
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    
    for url in voc_urls:
        filename = url.split('/')[-1]
        tar_path = data_path / filename
        
        # Download
        download_file(url, tar_path)
        
        # Extract
        extract_archive(tar_path, data_path)
        
        # Clean up
        tar_path.unlink()
    
    # Convert VOC to COCO format
    convert_voc_to_coco(data_dir)
    
    logger.info("Pascal VOC 2007 dataset downloaded and prepared")


def convert_voc_to_coco(data_dir):
    """Convert Pascal VOC format to COCO format"""
    logger.info("Converting VOC annotations to COCO format")
    
    voc_dir = Path(data_dir) / "VOCdevkit" / "VOC2007"
    if not voc_dir.exists():
        logger.warning("VOC directory not found, skipping conversion")
        return
    
    # This is a simplified conversion - you might want to use a more complete implementation
    # For now, create placeholder COCO annotations
    annotations_dir = voc_dir / "annotations"
    annotations_dir.mkdir(exist_ok=True)
    
    # Create minimal COCO annotations for VOC
    voc_categories = [
        {"id": 1, "name": "person", "supercategory": "person"},
        {"id": 2, "name": "car", "supercategory": "vehicle"},
        {"id": 3, "name": "bicycle", "supercategory": "vehicle"},
        # Add more VOC categories as needed
    ]
    
    # Create placeholder annotations
    for split in ["train", "val"]:
        coco_annotation = {
            "images": [],
            "annotations": [],
            "categories": voc_categories
        }
        
        coco_file = annotations_dir / f"instances_{split}.json"
        with open(coco_file, 'w') as f:
            json.dump(coco_annotation, f)


def prepare_dataset(dataset_name, data_dir):
    """Main function to prepare datasets"""
    logger.info(f"Preparing dataset: {dataset_name}")
    
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    
    if dataset_name.lower() == "coco_sample":
        create_coco_sample_dataset(data_dir)
    elif dataset_name.lower() == "balloon":
        download_balloon_dataset(data_dir)
    elif dataset_name.lower() == "voc2007":
        download_voc2007_dataset(data_dir)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    logger.info(f"Dataset {dataset_name} prepared successfully")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download and prepare datasets")
    parser.add_argument("--dataset", required=True, 
                       choices=["coco_sample", "balloon", "voc2007"],
                       help="Dataset to download")
    parser.add_argument("--data-dir", required=True,
                       help="Directory to save dataset")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    prepare_dataset(args.dataset, args.data_dir)
