#!/usr/bin/env python3
"""
Test script to validate MMDetection SageMaker setup
"""

import boto3
import sagemaker
import subprocess
import sys
import os
from pathlib import Path


def test_aws_credentials():
    """Test AWS credentials and permissions"""
    print("Testing AWS credentials...")
    
    try:
        sts = boto3.client('sts')
        identity = sts.get_caller_identity()
        print(f"✓ AWS credentials valid")
        print(f"  Account: {identity['Account']}")
        print(f"  User/Role: {identity['Arn']}")
        return True
    except Exception as e:
        print(f"✗ AWS credentials failed: {e}")
        return False


def test_sagemaker_permissions():
    """Test SageMaker permissions"""
    print("\nTesting SageMaker permissions...")
    
    try:
        session = sagemaker.Session()
        bucket = session.default_bucket()
        
        # Try to get execution role, if fails, use the one we created
        try:
            role = sagemaker.get_execution_role()
        except Exception:
            # Use the role we created
            role = "arn:aws:iam::897473508751:role/SageMakerExecutionRole-MMDetection"
            print(f"  Using created role: {role}")
        
        print(f"✓ SageMaker session created")
        print(f"  Role: {role}")
        print(f"  Bucket: {bucket}")
        return True
    except Exception as e:
        print(f"✗ SageMaker permissions failed: {e}")
        return False


def test_docker():
    """Test Docker installation"""
    print("\nTesting Docker...")
    
    try:
        result = subprocess.run(['docker', '--version'], 
                              capture_output=True, text=True, check=True)
        print(f"✓ Docker available: {result.stdout.strip()}")
        
        # Test Docker daemon
        result = subprocess.run(['docker', 'info'], 
                              capture_output=True, text=True, check=True)
        print("✓ Docker daemon running")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Docker failed: {e}")
        return False
    except FileNotFoundError:
        print("✗ Docker not installed")
        return False


def test_ecr_access():
    """Test ECR access"""
    print("\nTesting ECR access...")
    
    try:
        session = sagemaker.Session()
        region = session.boto_region_name
        
        # Test ECR login
        result = subprocess.run([
            'aws', 'ecr', 'get-login-password', '--region', region
        ], capture_output=True, text=True, check=True)
        
        print("✓ ECR login successful")
        
        # Test ECR repository access
        ecr = boto3.client('ecr')
        repos = ecr.describe_repositories()
        print(f"✓ ECR access verified ({len(repos['repositories'])} repositories)")
        return True
    except Exception as e:
        print(f"✗ ECR access failed: {e}")
        return False


def test_file_structure():
    """Test required files exist"""
    print("\nTesting file structure...")
    
    required_files = [
        'Dockerfile.training',
        'container_training/mmdetection_train.py',
        'container_training/dataset_utils.py',
        'quick_start.py',
        'config_examples.py'
    ]
    
    all_exist = True
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} missing")
            all_exist = False
    
    return all_exist


def test_python_dependencies():
    """Test Python dependencies"""
    print("\nTesting Python dependencies...")
    
    required_packages = [
        'boto3',
        'sagemaker',
    ]
    
    all_available = True
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} not installed")
            all_available = False
    
    return all_available


def test_container_build():
    """Test container build (optional)"""
    print("\nTesting container build (this may take a while)...")
    
    try:
        # Test if we can build the container
        result = subprocess.run([
            'docker', 'build', '-t', 'mmdetection-test', 
            '-f', 'Dockerfile.training', '.'
        ], capture_output=True, text=True, timeout=600)
        
        if result.returncode == 0:
            print("✓ Container build successful")
            
            # Clean up test image
            subprocess.run(['docker', 'rmi', 'mmdetection-test'], 
                         capture_output=True)
            return True
        else:
            print(f"✗ Container build failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("✗ Container build timed out (>10 minutes)")
        return False
    except Exception as e:
        print(f"✗ Container build error: {e}")
        return False


def run_quick_test():
    """Run a quick validation test"""
    print("\nRunning quick validation...")
    
    try:
        # Test dataset utility
        from container_training.dataset_utils import create_coco_sample_dataset
        import tempfile
        
        with tempfile.TemporaryDirectory() as temp_dir:
            create_coco_sample_dataset(temp_dir)
            
            # Check if files were created
            ann_dir = Path(temp_dir) / "annotations"
            if (ann_dir / "instances_train2017.json").exists():
                print("✓ Dataset utility working")
                return True
            else:
                print("✗ Dataset utility failed")
                return False
                
    except Exception as e:
        print(f"✗ Quick validation failed: {e}")
        return False


def main():
    """Run all tests"""
    print("=== MMDetection SageMaker Setup Test ===\n")
    
    tests = [
        ("AWS Credentials", test_aws_credentials),
        ("SageMaker Permissions", test_sagemaker_permissions),
        ("Docker", test_docker),
        ("ECR Access", test_ecr_access),
        ("File Structure", test_file_structure),
        ("Python Dependencies", test_python_dependencies),
        ("Quick Validation", run_quick_test),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Optional container build test
    if '--build-test' in sys.argv:
        results["Container Build"] = test_container_build()
    
    # Summary
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"{test_name:<25} {status}")
        if result:
            passed += 1
    
    print(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 All tests passed! Your setup is ready for MMDetection training.")
        print("\nNext steps:")
        print("1. Run: python quick_start.py --build-container --dry-run")
        print("2. Run: python quick_start.py --model faster_rcnn --dataset coco_sample")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please fix the issues above.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
