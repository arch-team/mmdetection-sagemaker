"""
Configuration examples for different MMDetection training scenarios
"""

# Example 1: Quick development training with COCO sample
QUICK_DEV_CONFIG = {
    'hyperparameters': {
        'config-file': 'faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py',
        'dataset': 'coco_sample',
        'download-dataset': True,
        'auto-scale': True,
        'validate': True,
        'options': 'train_cfg.max_epochs=5; default_hooks.checkpoint.interval=2; train_dataloader.batch_size=2'
    },
    'instance_type': 'ml.g4dn.xlarge',
    'instance_count': 1,
    'use_spot_instances': True,
    'max_run': 1800,  # 30 minutes
    'description': 'Quick development training for testing'
}

# Example 2: Production training with balloon dataset
BALLOON_PRODUCTION_CONFIG = {
    'hyperparameters': {
        'config-file': 'mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py',
        'dataset': 'balloon',
        'download-dataset': True,
        'auto-scale': True,
        'validate': True,
        'options': 'train_cfg.max_epochs=50; default_hooks.checkpoint.interval=10; optim_wrapper.optimizer.lr=0.001'
    },
    'instance_type': 'ml.g4dn.2xlarge',
    'instance_count': 1,
    'use_spot_instances': True,
    'max_run': 7200,  # 2 hours
    'description': 'Production training on balloon dataset with Mask R-CNN'
}

# Example 3: Distributed training with VOC dataset
VOC_DISTRIBUTED_CONFIG = {
    'hyperparameters': {
        'config-file': 'retinanet/retinanet_r50_fpn_1x_coco.py',
        'dataset': 'voc2007',
        'download-dataset': True,
        'auto-scale': True,
        'validate': True,
        'options': 'train_cfg.max_epochs=100; default_hooks.checkpoint.interval=20; train_dataloader.batch_size=4'
    },
    'instance_type': 'ml.g4dn.xlarge',
    'instance_count': 2,
    'distribution': {'pytorchddp': {'enabled': True}},
    'use_spot_instances': True,
    'max_run': 10800,  # 3 hours
    'description': 'Distributed training on Pascal VOC with RetinaNet'
}

# Example 4: High-performance training with YOLOX
YOLOX_HIGH_PERFORMANCE_CONFIG = {
    'hyperparameters': {
        'config-file': 'yolox/yolox_s_8x8_300e_coco.py',
        'dataset': 'coco_sample',
        'download-dataset': True,
        'auto-scale': True,
        'validate': True,
        'options': 'train_cfg.max_epochs=300; default_hooks.checkpoint.interval=50; train_dataloader.batch_size=8'
    },
    'instance_type': 'ml.p3.2xlarge',  # V100 GPU
    'instance_count': 1,
    'use_spot_instances': False,  # Use on-demand for reliability
    'max_run': 14400,  # 4 hours
    'description': 'High-performance YOLOX training with V100 GPU'
}

# Example 5: Custom model configuration
CUSTOM_FASTER_RCNN_CONFIG = {
    'hyperparameters': {
        'config-file': 'faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py',
        'dataset': 'balloon',
        'download-dataset': True,
        'auto-scale': False,  # Manual LR setting
        'validate': True,
        'options': '''
            train_cfg.max_epochs=30;
            default_hooks.checkpoint.interval=5;
            optim_wrapper.optimizer.lr=0.0025;
            optim_wrapper.optimizer.weight_decay=0.0001;
            train_dataloader.batch_size=4;
            model.roi_head.bbox_head.num_classes=1;
            model.rpn_head.anchor_generator.scales=[8];
            model.rpn_head.anchor_generator.ratios=[0.5,1.0,2.0]
        '''.replace('\n', ' ').strip()
    },
    'instance_type': 'ml.g4dn.xlarge',
    'instance_count': 1,
    'use_spot_instances': True,
    'max_run': 5400,
    'description': 'Custom Faster R-CNN configuration for single-class detection'
}

# Example 6: Multi-scale training configuration
MULTISCALE_TRAINING_CONFIG = {
    'hyperparameters': {
        'config-file': 'fcos/fcos_r50_caffe_fpn_gn-head_1x_coco.py',
        'dataset': 'coco_sample',
        'download-dataset': True,
        'auto-scale': True,
        'validate': True,
        'options': '''
            train_cfg.max_epochs=24;
            default_hooks.checkpoint.interval=6;
            train_pipeline[2].img_scale=[(1333,800),(1333,600),(1333,1000)];
            train_pipeline[2].multiscale_mode=range;
            train_dataloader.batch_size=2
        '''.replace('\n', ' ').strip()
    },
    'instance_type': 'ml.g4dn.2xlarge',
    'instance_count': 1,
    'use_spot_instances': True,
    'max_run': 7200,
    'description': 'Multi-scale training with FCOS'
}

# Configuration registry
CONFIGS = {
    'quick_dev': QUICK_DEV_CONFIG,
    'balloon_production': BALLOON_PRODUCTION_CONFIG,
    'voc_distributed': VOC_DISTRIBUTED_CONFIG,
    'yolox_high_perf': YOLOX_HIGH_PERFORMANCE_CONFIG,
    'custom_faster_rcnn': CUSTOM_FASTER_RCNN_CONFIG,
    'multiscale_fcos': MULTISCALE_TRAINING_CONFIG
}


def get_config(config_name):
    """Get configuration by name"""
    if config_name not in CONFIGS:
        available = ', '.join(CONFIGS.keys())
        raise ValueError(f"Config '{config_name}' not found. Available: {available}")
    
    return CONFIGS[config_name].copy()


def list_configs():
    """List all available configurations"""
    print("Available configurations:")
    for name, config in CONFIGS.items():
        print(f"  {name}: {config['description']}")


def create_estimator_from_config(config_name, image_uri, role, bucket):
    """Create SageMaker estimator from configuration"""
    import sagemaker
    from sagemaker.estimator import Estimator
    
    config = get_config(config_name)
    
    estimator_args = {
        'image_uri': image_uri,
        'role': role,
        'instance_count': config['instance_count'],
        'instance_type': config['instance_type'],
        'hyperparameters': config['hyperparameters'],
        'output_path': f's3://{bucket}/mmdetection-output/{config_name}',
        'base_job_name': f'mmdet-{config_name}',
        'use_spot_instances': config['use_spot_instances'],
        'max_run': config['max_run']
    }
    
    if config['use_spot_instances']:
        estimator_args['max_wait'] = config['max_run'] * 2
    
    if 'distribution' in config:
        estimator_args['distribution'] = config['distribution']
    
    return Estimator(**estimator_args)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Configuration examples")
    parser.add_argument("--list", action="store_true", help="List available configurations")
    parser.add_argument("--show", type=str, help="Show specific configuration")
    
    args = parser.parse_args()
    
    if args.list:
        list_configs()
    elif args.show:
        try:
            config = get_config(args.show)
            print(f"Configuration: {args.show}")
            print(f"Description: {config['description']}")
            print("\nHyperparameters:")
            for key, value in config['hyperparameters'].items():
                print(f"  {key}: {value}")
            print(f"\nInstance: {config['instance_type']} x {config['instance_count']}")
            print(f"Spot instances: {config['use_spot_instances']}")
            print(f"Max runtime: {config['max_run']} seconds")
        except ValueError as e:
            print(e)
    else:
        list_configs()
