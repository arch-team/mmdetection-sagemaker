# MMDetection Training on Amazon SageMaker

This repository provides a complete solution for training MMDetection models on Amazon SageMaker using public datasets. It supports automatic dataset download, distributed training, and multiple model architectures.

## Features

- **Automatic Dataset Download**: Supports COCO sample, Balloon, and Pascal VOC 2007 datasets
- **Multiple Model Architectures**: Faster R-CNN, Mask R-CNN, RetinaNet, FCOS, YOLOX
- **Distributed Training**: Multi-instance training with PyTorch DDP
- **Cost Optimization**: Spot instance support and configurable training duration
- **Easy Configuration**: Simple hyperparameter tuning and model selection

## Quick Start

### 1. Prerequisites

```bash
# Install required packages
pip install sagemaker boto3

# Configure AWS CLI
aws configure

# Ensure Docker is running
docker --version
```

### 2. Build and Push Container

```bash
# Build and push the training container
python quick_start.py --build-container --dry-run  # Preview configuration
python quick_start.py --build-container --model faster_rcnn --dataset coco_sample
```

### 3. Start Training

```bash
# Quick training with default settings
python quick_start.py --model faster_rcnn --dataset coco_sample --epochs 5

# Distributed training
python quick_start.py --model mask_rcnn --dataset balloon --epochs 10 --instance-count 2

# Custom instance type
python quick_start.py --model retinanet --dataset voc2007 --instance-type ml.g4dn.2xlarge
```

## Supported Configurations

### Models
- `faster_rcnn`: Faster R-CNN with ResNet-50 backbone
- `mask_rcnn`: Mask R-CNN with ResNet-50 backbone  
- `retinanet`: RetinaNet with ResNet-50 backbone
- `fcos`: FCOS with ResNet-50 backbone
- `yolox`: YOLOX-S model

### Datasets
- `coco_sample`: Small COCO dataset sample (for testing)
- `balloon`: Balloon dataset from Mask R-CNN paper
- `voc2007`: Pascal VOC 2007 dataset

### Instance Types
- `ml.g4dn.xlarge`: 1 GPU, 4 vCPUs, 16 GB RAM (recommended for single instance)
- `ml.g4dn.2xlarge`: 1 GPU, 8 vCPUs, 32 GB RAM
- `ml.g4dn.4xlarge`: 1 GPU, 16 vCPUs, 64 GB RAM
- `ml.p3.2xlarge`: 1 V100 GPU, 8 vCPUs, 61 GB RAM (for intensive training)

## Advanced Usage

### Using Jupyter Notebook

Open `mmdetection_public_datasets.ipynb` for interactive training:

```python
# Example: Custom training configuration
hyperparameters = {
    'config-file': 'mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py',
    'dataset': 'balloon',
    'download-dataset': True,
    'auto-scale': True,
    'validate': True,
    'options': 'train_cfg.max_epochs=20; default_hooks.checkpoint.interval=5; train_dataloader.batch_size=4'
}
```

### Custom Configuration Options

You can override any MMDetection configuration using the `options` parameter:

```python
# Learning rate and batch size
'options': 'optim_wrapper.optimizer.lr=0.001; train_dataloader.batch_size=8'

# Training epochs and validation
'options': 'train_cfg.max_epochs=50; default_hooks.checkpoint.interval=10'

# Model architecture changes
'options': 'model.backbone.depth=101; model.neck.num_outs=6'
```

### Distributed Training

For multi-instance training:

```python
estimator = Estimator(
    image_uri=image_uri,
    role=role,
    instance_count=2,  # Use 2 instances
    instance_type='ml.g4dn.xlarge',
    distribution={'pytorchddp': {'enabled': True}},  # Enable PyTorch DDP
    hyperparameters=hyperparameters
)
```

## File Structure

```
mmdetection-sagemaker/
├── Dockerfile.training              # Training container definition
├── container_training/
│   ├── mmdetection_train.py        # Main training script
│   └── dataset_utils.py            # Dataset download utilities
├── mmdetection_public_datasets.ipynb  # Interactive notebook
├── sagemaker_distributed.ipynb     # Original distributed training notebook
├── quick_start.py                  # Quick start script
├── build_and_push.sh              # Container build script
└── README.md                       # This file
```

## Training Process

1. **Container Initialization**: Downloads MMDetection and installs dependencies
2. **Dataset Preparation**: Automatically downloads and prepares the specified dataset
3. **Configuration**: Updates MMDetection config with SageMaker-specific settings
4. **Training**: Runs distributed training using PyTorch DDP if multiple instances
5. **Model Saving**: Saves trained model and config to S3 for later use

## Monitoring Training

### SageMaker Console
Monitor training progress in the SageMaker console under "Training jobs"

### AWS CLI
```bash
# Check training job status
aws sagemaker describe-training-job --training-job-name your-job-name

# View training logs
aws logs get-log-events --log-group-name /aws/sagemaker/TrainingJobs --log-stream-name your-job-name/algo-1-xxx
```

### Python SDK
```python
# Monitor training job
def monitor_training_job(job_name):
    sm_client = boto3.client('sagemaker')
    response = sm_client.describe_training_job(TrainingJobName=job_name)
    print(f"Status: {response['TrainingJobStatus']}")
    return response
```

## Cost Optimization

### Spot Instances
Use spot instances to reduce training costs by up to 90%:

```python
estimator = Estimator(
    # ... other parameters
    use_spot_instances=True,
    max_wait=7200,  # Maximum wait time for spot instances
    max_run=3600,   # Maximum training time
)
```

### Instance Selection
- Use `ml.g4dn.xlarge` for development and small datasets
- Use `ml.g4dn.2xlarge` or higher for production training
- Consider `ml.p3.2xlarge` for intensive training with large datasets

## Troubleshooting

### Common Issues

1. **Container Build Fails**
   ```bash
   # Ensure Docker is running and you have ECR permissions
   docker info
   aws ecr get-login-password --region us-east-1
   ```

2. **Training Job Fails**
   ```bash
   # Check CloudWatch logs for detailed error messages
   aws logs describe-log-groups --log-group-name-prefix /aws/sagemaker/TrainingJobs
   ```

3. **Out of Memory Errors**
   ```python
   # Reduce batch size in options
   'options': 'train_dataloader.batch_size=2; train_dataloader.num_workers=2'
   ```

4. **Dataset Download Issues**
   ```python
   # Use smaller datasets for testing
   hyperparameters['dataset'] = 'coco_sample'  # Instead of full COCO
   ```

### Debug Mode

Enable debug logging in the training script:

```python
hyperparameters['options'] = 'log_level=DEBUG; default_hooks.logger.interval=10'
```

## Examples

### Example 1: Quick Object Detection Training
```bash
python quick_start.py --model faster_rcnn --dataset coco_sample --epochs 5
```

### Example 2: Instance Segmentation with Custom Settings
```bash
python quick_start.py \
    --model mask_rcnn \
    --dataset balloon \
    --epochs 20 \
    --instance-type ml.g4dn.2xlarge \
    --no-spot
```

### Example 3: Distributed Training
```bash
python quick_start.py \
    --model yolox \
    --dataset voc2007 \
    --epochs 50 \
    --instance-count 2 \
    --instance-type ml.g4dn.xlarge
```

## Next Steps

1. **Custom Datasets**: Modify `dataset_utils.py` to support your own datasets
2. **Model Deployment**: Create inference container for model serving
3. **Hyperparameter Tuning**: Use SageMaker automatic model tuning
4. **MLOps Pipeline**: Set up continuous training with SageMaker Pipelines
5. **Model Registry**: Register trained models for version control

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review MMDetection documentation: https://mmdetection.readthedocs.io/
3. Check SageMaker documentation: https://docs.aws.amazon.com/sagemaker/

## License

This project is licensed under the Apache License 2.0 - see the MMDetection project for details.
