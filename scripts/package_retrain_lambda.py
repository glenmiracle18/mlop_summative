#!/usr/bin/env python3
"""
Package Retrain Lambda Function

This script packages the retrain Lambda function into a zip file and uploads it to S3.
It also creates a Lambda layer with the required dependencies.

Usage:
    python package_retrain_lambda.py --bucket your-bucket-name
Layers:
    numpy scikit-learn joblib

Author: Glen Miracle
"""

import os
import sys
import argparse
import shutil
import tempfile
import subprocess
import boto3
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Base directory of the project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def package_lambda_function(bucket_name):
    """Package the Lambda function and upload to S3"""
    logger.info("Packaging Lambda function")
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    logger.info(f"Using temporary directory: {temp_dir}")
    
    try:
        # Copy Lambda function to temp directory
        lambda_file = os.path.join(BASE_DIR, 'lambda', 'retrain_model_lambda.py')
        temp_lambda_file = os.path.join(temp_dir, 'retrain_model_lambda.py')
        shutil.copy2(lambda_file, temp_lambda_file)
        logger.info(f"Copied Lambda function to {temp_lambda_file}")
        
        # Create zip file
        zip_file = os.path.join(temp_dir, 'retrain_model_lambda.zip')
        logger.info(f"Creating zip file: {zip_file}")
        
        subprocess.run(
            f"cd {temp_dir} && zip -r retrain_model_lambda.zip retrain_model_lambda.py",
            shell=True,
            check=True
        )
        
        # Upload to S3
        s3_client = boto3.client('s3')
        s3_key = 'lambda/retrain_model_lambda.zip'
        logger.info(f"Uploading Lambda function to s3://{bucket_name}/{s3_key}")
        
        s3_client.upload_file(zip_file, bucket_name, s3_key)
        logger.info("Lambda function uploaded successfully")
        
        return f"s3://{bucket_name}/{s3_key}"
    
    finally:
        # Clean up
        logger.info(f"Cleaning up temporary directory: {temp_dir}")
        shutil.rmtree(temp_dir)

def create_layer(bucket_name):
    """Create and upload a Lambda layer with required dependencies"""
    logger.info("Creating Lambda layer with required dependencies")
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    logger.info(f"Using temporary directory: {temp_dir}")
    
    try:
        # Create directory structure for Lambda layer
        python_dir = os.path.join(temp_dir, 'python')
        os.makedirs(python_dir, exist_ok=True)
        
        # Install dependencies
        logger.info("Installing dependencies for Lambda layer")
        subprocess.run(
            f"python3 -m pip install -t {python_dir} numpy pandas scikit-learn joblib",
            shell=True,
            check=True
        )
        
        # Create zip file
        layer_zip = os.path.join(temp_dir, 'scipy_sklearn_layer.zip')
        logger.info(f"Creating layer zip file: {layer_zip}")
        
        subprocess.run(
            f"cd {temp_dir} && zip -r scipy_sklearn_layer.zip python",
            shell=True,
            check=True
        )
        
        # Upload to S3
        s3_client = boto3.client('s3')
        s3_key = 'layers/scipy_sklearn_layer.zip'
        logger.info(f"Uploading Lambda layer to s3://{bucket_name}/{s3_key}")
        
        s3_client.upload_file(layer_zip, bucket_name, s3_key)
        logger.info("Lambda layer uploaded successfully")
        
        return f"s3://{bucket_name}/{s3_key}"
    
    finally:
        # Clean up
        logger.info(f"Cleaning up temporary directory: {temp_dir}")
        shutil.rmtree(temp_dir)

def ensure_bucket_exists(bucket_name):
    """Ensure the S3 bucket exists, create it if it doesn't"""
    s3_client = boto3.client('s3')
    
    try:
        s3_client.head_bucket(Bucket=bucket_name)
        logger.info(f"Bucket {bucket_name} already exists")
    except Exception:
        logger.info(f"Creating bucket {bucket_name}")
        s3_client.create_bucket(Bucket=bucket_name)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Package Retrain Lambda Function')
    parser.add_argument('--bucket', required=True, help='S3 bucket to upload the Lambda package')
    
    args = parser.parse_args()
    
    try:
        # Ensure the bucket exists
        ensure_bucket_exists(args.bucket)
        
        # Create the Lambda layer
        layer_uri = create_layer(args.bucket)
        logger.info(f"Lambda layer uploaded to {layer_uri}")
        
        # Package and upload the Lambda function
        lambda_uri = package_lambda_function(args.bucket)
        logger.info(f"Lambda function uploaded to {lambda_uri}")
        
        # Print next steps
        print("\n===== Next Steps =====")
        print(f"1. Deploy the CloudFormation template using the following command:")
        print(f"   aws cloudformation deploy \\")
        print(f"     --template-file templates/retrain_lambda_template.yaml \\")
        print(f"     --stack-name spam-detection-retrain \\")
        print(f"     --capabilities CAPABILITY_IAM \\")
        print(f"     --parameter-overrides ModelBucketName={args.bucket}")
        
        return 0
    
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 