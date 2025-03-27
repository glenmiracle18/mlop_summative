#!/bin/bash

# Exit on error
set -e

echo "Creating Lambda deployment package..."

# Create package directory
PACKAGE_DIR="package"
mkdir -p $PACKAGE_DIR

# Copy the Lambda function code
echo "Copying Lambda function code..."
cp ../lambda/retrain_model_lambda.py $PACKAGE_DIR/
cp ../lambda/spam_detection_lambda.py $PACKAGE_DIR/

# Install dependencies
echo "Installing dependencies to package directory..."
pip3 install -r requirements.txt --target $PACKAGE_DIR

# Create zip file
echo "Creating zip file..."
cd $PACKAGE_DIR
zip -r ../lambda_deployment.zip .

echo "Deployment package created: lambda_deployment.zip" 