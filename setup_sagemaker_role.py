#!/usr/bin/env python3
"""
Script to create SageMaker execution role
"""

import boto3
import json
import time


def create_sagemaker_execution_role():
    """Create SageMaker execution role with necessary permissions"""
    
    iam = boto3.client('iam')
    role_name = 'SageMakerExecutionRole-MMDetection'
    
    # Trust policy for SageMaker
    trust_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {
                    "Service": "sagemaker.amazonaws.com"
                },
                "Action": "sts:AssumeRole"
            }
        ]
    }
    
    try:
        # Check if role already exists
        try:
            role = iam.get_role(RoleName=role_name)
            print(f"✓ Role {role_name} already exists")
            print(f"  ARN: {role['Role']['Arn']}")
            return role['Role']['Arn']
        except iam.exceptions.NoSuchEntityException:
            pass
        
        # Create the role
        print(f"Creating SageMaker execution role: {role_name}")
        response = iam.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(trust_policy),
            Description='SageMaker execution role for MMDetection training'
        )
        
        role_arn = response['Role']['Arn']
        print(f"✓ Role created: {role_arn}")
        
        # Attach AWS managed policies
        managed_policies = [
            'arn:aws:iam::aws:policy/AmazonSageMakerFullAccess',
            'arn:aws:iam::aws:policy/AmazonS3FullAccess',
            'arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryFullAccess'
        ]
        
        for policy_arn in managed_policies:
            iam.attach_role_policy(
                RoleName=role_name,
                PolicyArn=policy_arn
            )
            print(f"✓ Attached policy: {policy_arn}")
        
        # Wait for role to be available
        print("Waiting for role to be available...")
        time.sleep(10)
        
        return role_arn
        
    except Exception as e:
        print(f"✗ Error creating role: {e}")
        return None


def create_sagemaker_config(role_arn):
    """Create SageMaker config file"""
    import os
    from pathlib import Path
    
    config_dir = Path.home() / '.sagemaker'
    config_dir.mkdir(exist_ok=True)
    
    config_file = config_dir / 'config.yaml'
    
    config_content = f"""
SageMaker:
  PythonSDK:
    Modules:
      Session:
        default_bucket: null
      LocalSession:
        default_bucket: null
    default_role: {role_arn}
"""
    
    with open(config_file, 'w') as f:
        f.write(config_content.strip())
    
    print(f"✓ Created SageMaker config: {config_file}")
    print(f"  Default role: {role_arn}")


def main():
    print("=== Setting up SageMaker Execution Role ===\n")
    
    try:
        # Create the role
        role_arn = create_sagemaker_execution_role()
        
        if role_arn:
            # Create config file
            create_sagemaker_config(role_arn)
            
            print(f"\n✅ Setup completed successfully!")
            print(f"Role ARN: {role_arn}")
            print(f"\nYou can now use this role for SageMaker training jobs.")
            print(f"The role has been set as default in ~/.sagemaker/config.yaml")
            
        else:
            print("\n❌ Failed to create SageMaker execution role")
            return 1
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
