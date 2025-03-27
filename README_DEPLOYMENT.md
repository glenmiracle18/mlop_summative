# Spam Detection Model Retraining System - Deployment Guide

This guide provides step-by-step instructions for deploying the Spam Detection Model Retraining system using AWS CloudFormation.

## Prerequisites

- AWS CLI installed and configured with appropriate credentials
- Python 3.8 or newer
- Required Python packages: boto3

## Deployment Process

### 1. Prepare the Deployment Artifacts

Run the preparation script to create the necessary deployment artifacts:

```bash
# Install required dependencies
pip install boto3

# Run the preparation script
# Replace 'your-bucket-name' with a unique bucket name for your deployment
python scripts/prepare_deployment.py --bucket your-bucket-name
```

This script:
- Creates the specified S3 buckets if they don't exist
- Creates a Lambda deployment package with the retraining code
- Uploads the artifacts to S3
- Outputs the CloudFormation deployment command

### 2. Deploy the CloudFormation Stack

Use the command provided by the preparation script to deploy the CloudFormation stack:

```bash
aws cloudformation create-stack \
  --stack-name spam-detection-retrain \
  --template-body file://templates/retrain_lambda_template.yaml \
  --parameters ParameterKey=ModelBucketName,ParameterValue=your-bucket-name \
               ParameterKey=UploadBucketName,ParameterValue=spam-data-your-bucket-name \
  --capabilities CAPABILITY_IAM
```

The `CAPABILITY_IAM` flag is required because the template creates IAM roles.

### 3. Monitor the Deployment

You can monitor the deployment progress in the AWS CloudFormation console or using the CLI:

```bash
aws cloudformation describe-stacks --stack-name spam-detection-retrain
```

### 4. Testing the Deployment

Once the stack is deployed (status: CREATE_COMPLETE), you can test the system by uploading a CSV file to the uploads bucket:

```bash
# Example CSV file structure:
# message,label
# "Hello, how are you?",ham
# "Congratulations! You've won $1000",spam

# Upload a test file
aws s3 cp path/to/your/test-data.csv s3://spam-data-your-bucket-name/datasets/test-data.csv
```

The Lambda function will automatically trigger and:
1. Download the CSV file
2. Train a new spam detection model
3. Deploy a prediction API using Lambda and API Gateway

### 5. Monitoring and Logs

You can check the execution of the Lambda function in the AWS Lambda console or by viewing the CloudWatch logs:

```bash
# Get the most recent log stream
LOG_GROUP_NAME="/aws/lambda/spam-detection-retrain"
LOG_STREAM=$(aws logs describe-log-streams --log-group-name $LOG_GROUP_NAME --order-by LastEventTime --descending --limit 1 --query 'logStreams[0].logStreamName' --output text)

# Get the logs
aws logs get-log-events --log-group-name $LOG_GROUP_NAME --log-stream-name $LOG_STREAM
```

## Resources Created

The CloudFormation template creates the following resources:

- **IAM Roles**: 
  - Lambda execution role
- **Lambda Functions**: 
  - Retraining function
  - S3 notification configuration function
- **S3 Event Notifications**: 
  - Configures event notifications on the upload bucket to trigger the Lambda when a CSV file is uploaded

## Cleanup

To remove all resources created by this deployment:

```bash
# Delete the CloudFormation stack
aws cloudformation delete-stack --stack-name spam-detection-retrain

# Optionally, empty and delete the S3 buckets
aws s3 rm s3://your-bucket-name --recursive
aws s3 rb s3://your-bucket-name --force

aws s3 rm s3://spam-data-your-bucket-name --recursive
aws s3 rb s3://spam-data-your-bucket-name --force
```

## Troubleshooting

### Common Issues:

1. **Permission Issues**: 
   - Ensure your AWS user has sufficient permissions to create IAM roles and Lambda functions
   - Add the `--capabilities CAPABILITY_IAM` flag to the create-stack command

2. **S3 Bucket Already Exists**: 
   - If you get an error that the bucket already exists, you can continue with the deployment
   - The template now uses existing buckets instead of creating new ones

3. **Lambda Execution Issues**: 
   - Check CloudWatch logs for any execution errors
   - Common issues include permission problems, missing dependencies, or incorrect model paths

4. **S3 Notification Configuration**: 
   - If your Lambda isn't triggered by S3 uploads, check the bucket notification configuration
   - You may need to manually configure the S3 event notification in the AWS console 