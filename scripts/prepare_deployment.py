#!/usr/bin/env python3
"""
Prepare Deployment Script

This script prepares the necessary artifacts for CloudFormation deployment:
1. Creates a package for the Lambda function
2. Uploads it to S3
3. Ensures buckets exist

Usage:
  python prepare_deployment.py --bucket my-bucket-name [--data-bucket my-data-bucket]

Requirements:
  - AWS CLI configured with appropriate credentials
  - boto3 installed in your environment
"""

import argparse
import os
import shutil
import tempfile
import zipfile
import boto3

def create_lambda_package():
    """Package the Lambda function code"""
    print("Creating Lambda function package...")
    
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    lambda_package_path = os.path.join(temp_dir, 'retrain_model_lambda.zip')
    
    # Get the path to the lambda function
    lambda_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    lambda_file = os.path.join(lambda_dir, 'lambda', 'retrain_model_lambda.py')
    
    # Create a zip file with the Lambda function
    with zipfile.ZipFile(lambda_package_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(lambda_file, os.path.basename(lambda_file))
    
    print(f"Lambda package created at: {lambda_package_path}")
    return lambda_package_path

def ensure_bucket_exists(bucket_name, region=None):
    """Create a bucket if it doesn't exist"""
    s3_client = boto3.client('s3')
    try:
        s3_client.head_bucket(Bucket=bucket_name)
        print(f"Bucket {bucket_name} exists")
        return True
    except:
        print(f"Creating bucket {bucket_name}...")
        try:
            if region is None or region == 'us-east-1':
                s3_client.create_bucket(Bucket=bucket_name)
            else:
                s3_client.create_bucket(
                    Bucket=bucket_name,
                    CreateBucketConfiguration={'LocationConstraint': region}
                )
            
            # Enable versioning
            s3_client.put_bucket_versioning(
                Bucket=bucket_name,
                VersioningConfiguration={'Status': 'Enabled'}
            )
            
            return True
        except Exception as e:
            print(f"Error creating bucket: {str(e)}")
            return False

def upload_to_s3(bucket_name, lambda_package_path):
    """Upload the package to S3"""
    print(f"Uploading package to S3 bucket: {bucket_name}")
    
    s3_client = boto3.client('s3')
    
    # Upload Lambda package
    lambda_key = 'lambda/retrain_model_lambda.zip'
    print(f"Uploading Lambda package to s3://{bucket_name}/{lambda_key}")
    s3_client.upload_file(lambda_package_path, bucket_name, lambda_key)
    
    return {
        'lambda_key': lambda_key
    }

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Prepare deployment artifacts for CloudFormation')
    parser.add_argument('--bucket', required=True, help='S3 bucket name for deployment artifacts')
    parser.add_argument('--data-bucket', help='S3 bucket name for data uploads (default: spam-data-{bucket})')
    parser.add_argument('--region', default=None, help='AWS region for bucket creation')
    
    args = parser.parse_args()
    model_bucket_name = args.bucket
    data_bucket_name = args.data_bucket or f"spam-data-{model_bucket_name}"
    region = args.region
    
    # Get the AWS region if not specified
    if region is None:
        session = boto3.session.Session()
        region = session.region_name
    
    # Ensure buckets exist
    ensure_bucket_exists(model_bucket_name, region)
    ensure_bucket_exists(data_bucket_name, region)
    
    # Create and upload lambda package
    lambda_package_path = create_lambda_package()
    upload_info = upload_to_s3(model_bucket_name, lambda_package_path)
    
    print("\nDeployment preparation completed successfully!")
    print("CloudFormation deployment command:")
    print(f"aws cloudformation create-stack --stack-name spam-detection-retrain --template-body file://templates/retrain_lambda_template.yaml --parameters ParameterKey=ModelBucketName,ParameterValue={model_bucket_name} ParameterKey=UploadBucketName,ParameterValue={data_bucket_name} --capabilities CAPABILITY_IAM")
    
    # Cleanup
    print("Cleaning up temporary files...")
    if os.path.exists(os.path.dirname(lambda_package_path)):
        shutil.rmtree(os.path.dirname(lambda_package_path))

if __name__ == "__main__":
    main() 