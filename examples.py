#!/usr/bin/env python3
"""
Usage examples for MMDetection training on SageMaker
"""

import boto3
import sagemaker
from sagemaker.estimator import Estimator
import time
from config_examples import get_config, list_configs, create_estimator_from_config


def setup_sagemaker():
    """Initialize SageMaker session and get basic info"""
    session = sagemaker.Session()
    region = session.boto_region_name
    account = boto3.client('sts').get_caller_identity().get('Account')
    bucket = session.default_bucket()
    role = sagemaker.get_execution_role()
    
    return session, region, account, bucket, role


def example_1_quick_development():
    """Example 1: Quick development training"""
    print("=== Example 1: Quick Development Training ===")
    
    session, region, account, bucket, role = setup_sagemaker()
    image_uri = f"{account}.dkr.ecr.{region}.amazonaws.com/mmdetection-training:latest"
    
    # Use predefined quick development configuration
    estimator = create_estimator_from_config('quick_dev', image_uri, role, bucket)
    
    print("Configuration:")
    config = get_config('quick_dev')
    for key, value in config['hyperparameters'].items():
        print(f"  {key}: {value}")
    
    # Start training (uncomment to actually run)
    # job_name = f"mmdet-quick-dev-{int(time.time())}"
    # estimator.fit(job_name=job_name)
    # print(f"Training job started: {job_name}")
    
    print("To run this example, uncomment the fit() call above")
    return estimator


def example_2_custom_configuration():
    """Example 2: Custom configuration"""
    print("\n=== Example 2: Custom Configuration ===")
    
    session, region, account, bucket, role = setup_sagemaker()
    image_uri = f"{account}.dkr.ecr.{region}.amazonaws.com/mmdetection-training:latest"
    
    # Custom hyperparameters
    custom_hyperparameters = {
        'config-file': 'mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py',
        'dataset': 'balloon',
        'download-dataset': True,
        'auto-scale': False,  # Manual learning rate
        'validate': True,
        'options': '''
            train_cfg.max_epochs=25;
            default_hooks.checkpoint.interval=5;
            optim_wrapper.optimizer.lr=0.002;
            optim_wrapper.optimizer.momentum=0.9;
            optim_wrapper.optimizer.weight_decay=0.0001;
            train_dataloader.batch_size=2;
            train_dataloader.num_workers=2;
            model.roi_head.bbox_head.num_classes=1;
            model.roi_head.mask_head.num_classes=1
        '''.replace('\n', ' ').strip()
    }
    
    estimator = Estimator(
        image_uri=image_uri,
        role=role,
        instance_count=1,
        instance_type='ml.g4dn.xlarge',
        hyperparameters=custom_hyperparameters,
        output_path=f's3://{bucket}/mmdetection-custom',
        base_job_name='mmdet-custom',
        use_spot_instances=True,
        max_wait=7200,
        max_run=3600,
    )
    
    print("Custom configuration created for single-class balloon detection")
    return estimator


def example_3_distributed_training():
    """Example 3: Distributed training"""
    print("\n=== Example 3: Distributed Training ===")
    
    session, region, account, bucket, role = setup_sagemaker()
    image_uri = f"{account}.dkr.ecr.{region}.amazonaws.com/mmdetection-training:latest"
    
    # Use distributed configuration
    estimator = create_estimator_from_config('voc_distributed', image_uri, role, bucket)
    
    print("Distributed training configuration:")
    config = get_config('voc_distributed')
    print(f"  Instances: {config['instance_count']} x {config['instance_type']}")
    print(f"  Dataset: {config['hyperparameters']['dataset']}")
    print(f"  Model: {config['hyperparameters']['config-file']}")
    
    return estimator


def example_4_hyperparameter_comparison():
    """Example 4: Compare different hyperparameter settings"""
    print("\n=== Example 4: Hyperparameter Comparison ===")
    
    session, region, account, bucket, role = setup_sagemaker()
    image_uri = f"{account}.dkr.ecr.{region}.amazonaws.com/mmdetection-training:latest"
    
    # Different learning rate configurations
    lr_configs = [
        {'lr': 0.001, 'name': 'low-lr'},
        {'lr': 0.01, 'name': 'high-lr'},
        {'lr': 0.005, 'name': 'medium-lr'}
    ]
    
    estimators = []
    
    for config in lr_configs:
        hyperparameters = {
            'config-file': 'faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py',
            'dataset': 'coco_sample',
            'download-dataset': True,
            'auto-scale': False,
            'validate': True,
            'options': f'train_cfg.max_epochs=10; optim_wrapper.optimizer.lr={config["lr"]}'
        }
        
        estimator = Estimator(
            image_uri=image_uri,
            role=role,
            instance_count=1,
            instance_type='ml.g4dn.xlarge',
            hyperparameters=hyperparameters,
            output_path=f's3://{bucket}/mmdetection-lr-comparison/{config["name"]}',
            base_job_name=f'mmdet-{config["name"]}',
            use_spot_instances=True,
            max_wait=3600,
            max_run=1800,
        )
        
        estimators.append((config['name'], estimator))
        print(f"Created estimator for LR={config['lr']}")
    
    return estimators


def example_5_model_comparison():
    """Example 5: Compare different model architectures"""
    print("\n=== Example 5: Model Architecture Comparison ===")
    
    session, region, account, bucket, role = setup_sagemaker()
    image_uri = f"{account}.dkr.ecr.{region}.amazonaws.com/mmdetection-training:latest"
    
    # Different model configurations
    model_configs = [
        {
            'name': 'faster-rcnn',
            'config': 'faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py',
            'description': 'Two-stage detector with RPN'
        },
        {
            'name': 'retinanet',
            'config': 'retinanet/retinanet_r50_fpn_1x_coco.py',
            'description': 'Single-stage detector with focal loss'
        },
        {
            'name': 'fcos',
            'config': 'fcos/fcos_r50_caffe_fpn_gn-head_1x_coco.py',
            'description': 'Anchor-free single-stage detector'
        }
    ]
    
    estimators = []
    
    for model in model_configs:
        hyperparameters = {
            'config-file': model['config'],
            'dataset': 'coco_sample',
            'download-dataset': True,
            'auto-scale': True,
            'validate': True,
            'options': 'train_cfg.max_epochs=15; default_hooks.checkpoint.interval=5'
        }
        
        estimator = Estimator(
            image_uri=image_uri,
            role=role,
            instance_count=1,
            instance_type='ml.g4dn.xlarge',
            hyperparameters=hyperparameters,
            output_path=f's3://{bucket}/mmdetection-model-comparison/{model["name"]}',
            base_job_name=f'mmdet-{model["name"]}',
            use_spot_instances=True,
            max_wait=5400,
            max_run=3600,
        )
        
        estimators.append((model['name'], estimator, model['description']))
        print(f"Created estimator for {model['name']}: {model['description']}")
    
    return estimators


def example_6_batch_training():
    """Example 6: Run multiple training jobs"""
    print("\n=== Example 6: Batch Training ===")
    
    # This example shows how to run multiple training jobs in sequence
    configs_to_run = ['quick_dev', 'balloon_production']
    
    session, region, account, bucket, role = setup_sagemaker()
    image_uri = f"{account}.dkr.ecr.{region}.amazonaws.com/mmdetection-training:latest"
    
    job_names = []
    
    for config_name in configs_to_run:
        estimator = create_estimator_from_config(config_name, image_uri, role, bucket)
        job_name = f"mmdet-batch-{config_name}-{int(time.time())}"
        
        print(f"Prepared job: {job_name}")
        job_names.append(job_name)
        
        # To actually run the jobs, uncomment the following:
        # estimator.fit(job_name=job_name, wait=False)  # Don't wait for completion
        # time.sleep(10)  # Small delay between job submissions
    
    print("To run batch training, uncomment the fit() calls above")
    return job_names


def monitor_training_jobs(job_names):
    """Monitor multiple training jobs"""
    print(f"\n=== Monitoring {len(job_names)} Training Jobs ===")
    
    sm_client = boto3.client('sagemaker')
    
    for job_name in job_names:
        try:
            response = sm_client.describe_training_job(TrainingJobName=job_name)
            status = response['TrainingJobStatus']
            print(f"{job_name}: {status}")
            
            if status == 'Completed':
                training_time = response.get('TrainingTimeInSeconds', 0)
                print(f"  Training time: {training_time} seconds")
                print(f"  Model artifacts: {response['ModelArtifacts']['S3ModelArtifacts']}")
            elif status == 'Failed':
                print(f"  Failure reason: {response.get('FailureReason', 'Unknown')}")
                
        except Exception as e:
            print(f"{job_name}: Error - {e}")


def main():
    """Run examples"""
    print("MMDetection SageMaker Training Examples")
    print("="*50)
    
    # List available configurations
    print("\nAvailable predefined configurations:")
    list_configs()
    
    # Run examples
    examples = [
        example_1_quick_development,
        example_2_custom_configuration,
        example_3_distributed_training,
        example_4_hyperparameter_comparison,
        example_5_model_comparison,
        example_6_batch_training
    ]
    
    for example_func in examples:
        try:
            example_func()
        except Exception as e:
            print(f"Error in {example_func.__name__}: {e}")
    
    print("\n" + "="*50)
    print("Examples completed!")
    print("\nTo actually run training jobs:")
    print("1. Uncomment the estimator.fit() calls in the examples")
    print("2. Or use: python quick_start.py --model <model> --dataset <dataset>")
    print("3. Or use the Jupyter notebook: mmdetection_public_datasets.ipynb")


if __name__ == "__main__":
    main()
