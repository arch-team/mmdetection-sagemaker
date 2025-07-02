from argparse import ArgumentParser
import os
from mmengine import Config
import json
import subprocess
import sys
import shutil
import logging
import urllib.request
import zipfile
import tarfile

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_training_world():
    """
    Calculates number of devices in Sagemaker distributed cluster
    """
    # Get params of Sagemaker distributed cluster from predefined env variables
    num_gpus = int(os.environ["SM_NUM_GPUS"])
    num_cpus = int(os.environ["SM_NUM_CPUS"])
    hosts = json.loads(os.environ["SM_HOSTS"])
    current_host = os.environ["SM_CURRENT_HOST"]

    # Define PyTorch training world
    world = {}
    world["number_of_processes"] = num_gpus if num_gpus > 0 else num_cpus
    world["number_of_machines"] = len(hosts)
    world["size"] = world["number_of_processes"] * world["number_of_machines"]
    world["machine_rank"] = hosts.index(current_host)
    world["master_addr"] = hosts[0]
    world["master_port"] = "55555"  # port is defined by Sagemaker

    return world


def download_public_dataset(dataset_name, data_dir):
    """
    Download public datasets for training
    """
    logger.info(f"Downloading {dataset_name} dataset to {data_dir}")
    
    if dataset_name.lower() == "coco_sample":
        # Download a small sample of COCO dataset for quick testing
        download_coco_sample(data_dir)
    elif dataset_name.lower() == "balloon":
        # Download balloon dataset (small dataset for testing)
        download_balloon_dataset(data_dir)
    elif dataset_name.lower() == "voc2007":
        # Download Pascal VOC 2007 dataset
        download_voc2007_dataset(data_dir)
    else:
        raise ValueError(f"Dataset {dataset_name} is not supported for automatic download")


def download_coco_sample(data_dir):
    """Download a small sample of COCO dataset"""
    os.makedirs(data_dir, exist_ok=True)
    
    # Create sample COCO structure
    annotations_dir = os.path.join(data_dir, "annotations")
    train_dir = os.path.join(data_dir, "train2017")
    val_dir = os.path.join(data_dir, "val2017")
    
    os.makedirs(annotations_dir, exist_ok=True)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # Download sample annotations (you would need to provide actual URLs)
    logger.info("Creating sample COCO dataset structure...")
    
    # For demonstration, create minimal annotation files
    create_sample_coco_annotations(annotations_dir)


def download_balloon_dataset(data_dir):
    """Download balloon dataset from VIA tool"""
    balloon_url = "https://github.com/matterport/Mask_RCNN/releases/download/v2.1/balloon_dataset.zip"
    zip_path = os.path.join(data_dir, "balloon_dataset.zip")
    
    logger.info(f"Downloading balloon dataset from {balloon_url}")
    urllib.request.urlretrieve(balloon_url, zip_path)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)
    
    os.remove(zip_path)
    logger.info("Balloon dataset downloaded and extracted")


def download_voc2007_dataset(data_dir):
    """Download Pascal VOC 2007 dataset"""
    voc_urls = [
        "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar",
        "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar"
    ]
    
    for url in voc_urls:
        filename = url.split('/')[-1]
        tar_path = os.path.join(data_dir, filename)
        
        logger.info(f"Downloading {filename}")
        urllib.request.urlretrieve(url, tar_path)
        
        with tarfile.open(tar_path, 'r') as tar_ref:
            tar_ref.extractall(data_dir)
        
        os.remove(tar_path)
    
    logger.info("VOC 2007 dataset downloaded and extracted")


def create_sample_coco_annotations(annotations_dir):
    """Create minimal COCO annotation files for testing"""
    import json
    
    # Minimal COCO annotation structure
    sample_annotation = {
        "images": [
            {
                "id": 1,
                "width": 640,
                "height": 480,
                "file_name": "sample_001.jpg"
            }
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [100, 100, 200, 200],
                "area": 40000,
                "iscrowd": 0
            }
        ],
        "categories": [
            {
                "id": 1,
                "name": "person",
                "supercategory": "person"
            }
        ]
    }
    
    # Save train and val annotations
    for split in ["train2017", "val2017"]:
        ann_file = os.path.join(annotations_dir, f"instances_{split}.json")
        with open(ann_file, 'w') as f:
            json.dump(sample_annotation, f)
    
    logger.info("Sample COCO annotations created")


def training_configurator(args, world):
    """
    Configure training process by updating config file
    """
    # Get config file path
    if args.config_file.startswith('/'):
        abs_config_path = args.config_file
    else:
        abs_config_path = os.path.join("/opt/ml/code/mmdetection/configs", args.config_file)
    
    logger.info(f"Loading config from: {abs_config_path}")
    cfg = Config.fromfile(abs_config_path)
    
    # Handle different dataset types
    if args.dataset.lower() in ["coco", "coco_sample"]:
        configure_coco_dataset(cfg, args)
    elif args.dataset.lower() == "balloon":
        configure_balloon_dataset(cfg, args)
    elif args.dataset.lower() == "voc2007":
        configure_voc_dataset(cfg, args)
    else:
        raise NotImplementedError(f"Dataset {args.dataset} is not implemented")
    
    # Apply user options
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    
    # Auto-scale learning rate if requested
    if args.auto_scale:
        cfg = auto_scale_config(cfg, world)
    
    # Set work directory
    cfg.work_dir = os.environ.get('SM_OUTPUT_DATA_DIR', './work_dirs')
    
    # Save updated config
    updated_config = os.path.join(os.getcwd(), "updated_config.py")
    cfg.dump(updated_config)
    logger.info(f"Updated config saved to: {updated_config}")
    
    return updated_config


def configure_coco_dataset(cfg, args):
    """Configure COCO dataset paths"""
    data_root = os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training")
    
    # Download dataset if needed
    if args.download_dataset:
        download_public_dataset(args.dataset, data_root)
    
    cfg.data_root = data_root
    
    # Update dataset configurations
    if hasattr(cfg, 'train_dataloader'):
        cfg.train_dataloader.dataset.data_root = data_root
        cfg.train_dataloader.dataset.ann_file = 'annotations/instances_train2017.json'
        cfg.train_dataloader.dataset.data_prefix = dict(img='train2017/')
    
    if hasattr(cfg, 'val_dataloader'):
        cfg.val_dataloader.dataset.data_root = data_root
        cfg.val_dataloader.dataset.ann_file = 'annotations/instances_val2017.json'
        cfg.val_dataloader.dataset.data_prefix = dict(img='val2017/')
    
    if hasattr(cfg, 'test_dataloader'):
        cfg.test_dataloader.dataset.data_root = data_root
        cfg.test_dataloader.dataset.ann_file = 'annotations/instances_val2017.json'
        cfg.test_dataloader.dataset.data_prefix = dict(img='val2017/')


def configure_balloon_dataset(cfg, args):
    """Configure balloon dataset paths"""
    data_root = os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training")
    
    if args.download_dataset:
        download_public_dataset(args.dataset, data_root)
    
    # This would require custom dataset configuration for balloon dataset
    # You would need to create appropriate dataset configs
    logger.warning("Balloon dataset configuration needs custom implementation")


def configure_voc_dataset(cfg, args):
    """Configure Pascal VOC dataset paths"""
    data_root = os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training")
    
    if args.download_dataset:
        download_public_dataset(args.dataset, data_root)
    
    # Configure VOC dataset paths
    cfg.data_root = data_root
    # VOC configuration would need specific setup based on MMDetection VOC configs


def auto_scale_config(cfg, world):
    """
    Automatically scale learning rate based on number of processes
    """
    old_world_size = 8  # Default world size for MMDetection configs
    scale = world["size"] / old_world_size
    
    # Scale learning rate
    if hasattr(cfg, 'optim_wrapper') and hasattr(cfg.optim_wrapper, 'optimizer'):
        old_lr = cfg.optim_wrapper.optimizer.lr
        cfg.optim_wrapper.optimizer.lr = old_lr * scale
        logger.info(f"Scaled learning rate from {old_lr} to {cfg.optim_wrapper.optimizer.lr}")
    
    # Scale warmup iterations
    if hasattr(cfg, 'param_scheduler'):
        for scheduler in cfg.param_scheduler:
            if hasattr(scheduler, 'warmup_iters'):
                old_warmup = scheduler.warmup_iters
                scheduler.warmup_iters = int(old_warmup / scale)
                logger.info(f"Scaled warmup iterations from {old_warmup} to {scheduler.warmup_iters}")
    
    return cfg


def options_to_dict(options):
    """
    Convert options string to dictionary
    """
    options_dict = dict(item.split("=") for item in options.split("; "))
    
    for key, value in options_dict.items():
        value = [_parse_int_float_bool(v) for v in value.split(",")]
        if len(value) == 1:
            value = value[0]
        options_dict[key] = value
    return options_dict


def _parse_int_float_bool(val):
    """Parse string value to appropriate type"""
    try:
        return int(val)
    except ValueError:
        pass
    try:
        return float(val)
    except ValueError:
        pass
    if val.lower() in ['true', 'false']:
        return True if val.lower() == 'true' else False
    return val


def save_model(config_path, work_dir, model_dir):
    """
    Save model artifacts to SageMaker model directory
    """
    logger.info(f"Saving model artifacts from {work_dir} to {model_dir}")
    
    # Create model directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Copy config file
    try:
        new_config_path = os.path.join(model_dir, "config.py")
        shutil.copyfile(config_path, new_config_path)
        logger.info(f"Config saved to {new_config_path}")
    except Exception as e:
        logger.error(f"Failed to copy config: {e}")
    
    # Copy model checkpoints
    if os.path.exists(work_dir):
        for file in os.listdir(work_dir):
            if file.endswith(".pth"):
                try:
                    checkpoint_path = os.path.join(work_dir, file)
                    new_checkpoint_path = os.path.join(model_dir, file)
                    shutil.copyfile(checkpoint_path, new_checkpoint_path)
                    logger.info(f"Checkpoint {file} saved to model directory")
                except Exception as e:
                    logger.error(f"Failed to copy checkpoint {file}: {e}")
    
    logger.info(f"Model artifacts saved to {model_dir}")


if __name__ == "__main__":
    logger.info('Starting MMDetection training...')
    
    parser = ArgumentParser()
    parser.add_argument('--config-file', type=str, required=True,
                        help="MMDetection config file path (relative to configs/ or absolute path)")
    parser.add_argument('--dataset', type=str, default="coco",
                        choices=["coco", "coco_sample", "balloon", "voc2007"],
                        help="Dataset type to use")
    parser.add_argument('--download-dataset', action='store_true',
                        help="Download public dataset automatically")
    parser.add_argument('--options', nargs='+', type=str, default=None,
                        help='Config overrides in format "key1=value1; key2=value2"')
    parser.add_argument('--auto-scale', action='store_true',
                        help="Auto-scale learning rate based on cluster size")
    parser.add_argument('--validate', action='store_true',
                        help="Run validation during training")
    
    args, unknown = parser.parse_known_args()
    
    if args.options is not None:
        args.options = options_to_dict(args.options[0])
    
    if unknown:
        logger.warning(f"Unknown arguments: {unknown}")
    
    # Get distributed training configuration
    world = get_training_world()
    logger.info(f"Training world: {world}")
    
    # Configure training
    config_file = training_configurator(args, world)
    
    # Prepare training command
    if world['size'] > 1:
        # Distributed training
        launch_config = [
            "python", "-m", "torch.distributed.launch",
            "--nnodes", str(world['number_of_machines']),
            "--node_rank", str(world['machine_rank']),
            "--nproc_per_node", str(world['number_of_processes']),
            "--master_addr", world['master_addr'],
            "--master_port", world['master_port']
        ]
    else:
        # Single node training
        launch_config = ["python"]
    
    train_config = [
        os.path.join(os.environ["MMDETECTION"], "tools/train.py"),
        config_file,
        "--work-dir", os.environ.get('SM_OUTPUT_DATA_DIR', './work_dirs')
    ]
    
    if world['size'] > 1:
        train_config.extend(["--launcher", "pytorch"])
    
    if not args.validate:
        train_config.append("--cfg-options")
        train_config.append("val_cfg=None")
        train_config.append("val_dataloader=None")
        train_config.append("val_evaluator=None")
    
    # Execute training
    joint_cmd = " ".join(launch_config + train_config)
    logger.info(f"Executing command: {joint_cmd}")
    
    process = subprocess.Popen(
        joint_cmd,
        stderr=subprocess.STDOUT,
        stdout=subprocess.PIPE,
        shell=True
    )
    
    while True:
        output = process.stdout.readline()
        if process.poll() is not None:
            break
        if output:
            print(output.decode("utf-8").strip())
    
    rc = process.poll()
    
    if process.returncode != 0:
        raise subprocess.CalledProcessError(
            returncode=process.returncode,
            cmd=joint_cmd
        )
    
    # Save model artifacts
    save_model(
        config_file,
        os.environ.get('SM_OUTPUT_DATA_DIR', './work_dirs'),
        os.environ.get('SM_MODEL_DIR', './model')
    )
    
    logger.info('Training completed successfully!')
    sys.exit(process.returncode)
    
