#!/usr/bin/env python3
"""
Quick start script for MMDetection training on SageMaker
"""

import boto3
import sagemaker
from sagemaker.estimator import Estimator
import time
import argparse
import json


def get_sagemaker_session():
    """Initialize SageMaker session"""
    session = sagemaker.Session()
    region = session.boto_region_name
    account = boto3.client('sts').get_caller_identity().get('Account')
    bucket = session.default_bucket()
    
    # Try to get execution role, if fails, use the one we created
    try:
        role = sagemaker.get_execution_role()
    except Exception:
        # Use the role we created
        role = f"arn:aws:iam::{account}:role/SageMakerExecutionRole-MMDetection"
    
    return session, region, account, bucket, role


def build_and_push_container(container_name, region, account):
    """Build and push training container"""
    import subprocess
    
    image_uri = f"{account}.dkr.ecr.{region}.amazonaws.com/{container_name}:latest"
    
    print("Building and pushing container...")
    
    # Login to ECR
    subprocess.run([
        "aws", "ecr", "get-login-password", "--region", region
    ], check=True, capture_output=True, text=True)
    
    # Create repository if it doesn't exist
    try:
        subprocess.run([
            "aws", "ecr", "describe-repositories", 
            "--repository-names", container_name, 
            "--region", region
        ], check=True, capture_output=True)
    except subprocess.CalledProcessError:
        subprocess.run([
            "aws", "ecr", "create-repository", 
            "--repository-name", container_name, 
            "--region", region
        ], check=True)
    
    # Build and push
    subprocess.run([
        "docker", "build", "-t", f"{container_name}:latest", 
        "-f", "Dockerfile.training", "."
    ], check=True)
    
    subprocess.run([
        "docker", "tag", f"{container_name}:latest", image_uri
    ], check=True)
    
    subprocess.run([
        "docker", "push", image_uri
    ], check=True)
    
    print(f"Container pushed to: {image_uri}")
    return image_uri


def create_training_job(
    image_uri, 
    role, 
    bucket,
    config_file,
    dataset,
    instance_type="ml.g4dn.xlarge",
    instance_count=1,
    max_epochs=5,
    use_spot=True
):
    """Create and start training job"""
    
    hyperparameters = {
        'config-file': config_file,
        'dataset': dataset,
        'download-dataset': True,
        'auto-scale': True,
        'validate': True,
        'options': f'train_cfg.max_epochs={max_epochs}; default_hooks.checkpoint.interval={max(1, max_epochs//5)}'
    }
    
    estimator = Estimator(
        image_uri=image_uri,
        role=role,
        instance_count=instance_count,
        instance_type=instance_type,
        hyperparameters=hyperparameters,
        output_path=f's3://{bucket}/mmdetection-output',
        base_job_name='mmdetection-quickstart',
        use_spot_instances=use_spot,
        max_wait=7200 if use_spot else None,
        max_run=3600,
    )
    
    if instance_count > 1:
        estimator.distribution = {'pytorchddp': {'enabled': True}}
    
    job_name = f"mmdetection-{dataset}-{int(time.time())}"
    print(f"Starting training job: {job_name}")
    
    estimator.fit(job_name=job_name)
    
    return estimator, job_name


def main():
    parser = argparse.ArgumentParser(description="Quick start MMDetection training")
    
    # Model and dataset options
    parser.add_argument("--model", default="faster_rcnn", 
                       choices=["faster_rcnn", "mask_rcnn", "retinanet", "fcos", "yolox"],
                       help="Model architecture to use")
    parser.add_argument("--dataset", default="coco_sample",
                       choices=["coco_sample", "balloon", "voc2007"],
                       help="Dataset to use")
    
    # Training options
    parser.add_argument("--epochs", type=int, default=5,
                       help="Number of training epochs")
    parser.add_argument("--instance-type", default="ml.g4dn.xlarge",
                       help="SageMaker instance type")
    parser.add_argument("--instance-count", type=int, default=1,
                       help="Number of instances for distributed training")
    
    # Container options
    parser.add_argument("--container-name", default="mmdetection-training",
                       help="ECR container name")
    parser.add_argument("--build-container", action="store_true",
                       help="Build and push container before training")
    
    # Other options
    parser.add_argument("--no-spot", action="store_true",
                       help="Don't use spot instances")
    parser.add_argument("--dry-run", action="store_true",
                       help="Print configuration without starting training")
    
    args = parser.parse_args()
    
    # Model configuration mapping
    model_configs = {
        "faster_rcnn": "faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py",
        "mask_rcnn": "mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py",
        "retinanet": "retinanet/retinanet_r50_fpn_1x_coco.py",
        "fcos": "fcos/fcos_r50_caffe_fpn_gn-head_1x_coco.py",
        "yolox": "yolox/yolox_s_8x8_300e_coco.py"
    }
    
    config_file = model_configs[args.model]
    
    # Initialize SageMaker
    session, region, account, bucket, role = get_sagemaker_session()
    
    print("=== MMDetection Quick Start ===")
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Config: {config_file}")
    print(f"Epochs: {args.epochs}")
    print(f"Instance: {args.instance_type} x {args.instance_count}")
    print(f"Region: {region}")
    print(f"Bucket: {bucket}")
    print()
    
    if args.dry_run:
        print("Dry run mode - configuration printed above")
        return
    
    # Build container if requested
    if args.build_container:
        image_uri = build_and_push_container(args.container_name, region, account)
    else:
        image_uri = f"{account}.dkr.ecr.{region}.amazonaws.com/{args.container_name}:latest"
        print(f"Using existing container: {image_uri}")
    
    # Start training
    try:
        estimator, job_name = create_training_job(
            image_uri=image_uri,
            role=role,
            bucket=bucket,
            config_file=config_file,
            dataset=args.dataset,
            instance_type=args.instance_type,
            instance_count=args.instance_count,
            max_epochs=args.epochs,
            use_spot=not args.no_spot
        )
        
        print(f"Training job '{job_name}' started successfully!")
        print(f"Monitor progress in SageMaker console or use:")
        print(f"aws sagemaker describe-training-job --training-job-name {job_name}")
        
    except Exception as e:
        print(f"Error starting training job: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
