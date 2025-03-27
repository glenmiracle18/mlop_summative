# Spam Detection Model - Complete Deployment Guide
## Author: Bonyu Miracle Glen
![MLOps Architecture](https://i.imgur.com/lmT7w2j.png)

## Overview

This comprehensive guide documents the Spam Detection Model system - a serverless machine learning application built on AWS that enables users to detect spam messages and retrain the model with new data. The system follows MLOps best practices, providing continuous model improvement while maintaining cost efficiency through serverless architecture.

## Technology Stack

### Cloud Infrastructure
- **AWS Lambda** - Serverless compute for model prediction and retraining
- **Amazon S3** - Storage for model artifacts and training datasets
- **Amazon API Gateway** - RESTful API creation for model inference
- **AWS CloudFormation** - Infrastructure as Code (IaC) for deployment
- **AWS CloudWatch** - Monitoring and logging
- **AWS IAM** - Identity and access management

### Application Components
- **⚛️Frontend**: Next.js application running in Docker
- **🐍Backend**: Python-based Lambda functions
- **🤖Machine Learning**: scikit-learn for model training and prediction
- **💽Data Processing**: pandas and numpy for data manipulation
- **🐳Containerization**: Docker for frontend deployment

### Key Features
- Serverless architecture (pay-per-use)
- CI/CD integration through CloudFormation
- Model retraining capabilities
- Real-time spam detection API
- Modern React frontend
- Comprehensive monitoring and logging

## System Architecture

The system follows a serverless architecture pattern where:

1. Users interact with a React frontend to submit prediction requests or upload new training data
2. S3 buckets store training data and model artifacts
3. Lambda functions handle model prediction and retraining
4. API Gateway exposes the model prediction capabilities as REST endpoints
5. CloudWatch provides monitoring and logging for the entire system





## Deployment Instructions

### Prerequisites
- AWS CLI installed and configured with appropriate permissions
- Python 3.8 or newer
- Docker and Docker Compose
- Git (for version control)

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/spam-detection-model.git
cd spam-detection-model
```

### Step 2: Prepare AWS Resources

1. Install required Python dependencies:
```bash
pip install boto3 scikit-learn pandas numpy
```

2. Create the S3 buckets and prepare deployment artifacts:
```bash
# Create a unique bucket name
BUCKET_NAME="spam-detection-model-$(date +%s)"

# Run the preparation script
python scripts/prepare_deployment.py --bucket $BUCKET_NAME
```

This script will:
- Create two S3 buckets (model artifacts and data uploads)
- Package the Lambda functions
- Upload necessary artifacts to S3
- Output the CloudFormation deployment command

### Step 3: Deploy CloudFormation Stack

Use the command provided by the preparation script to deploy the CloudFormation stack:

```bash
aws cloudformation create-stack \
  --stack-name spam-detection-retrain \
  --template-body file://templates/retrain_lambda_template.yaml \
  --parameters ParameterKey=ModelBucketName,ParameterValue=$BUCKET_NAME \
               ParameterKey=UploadBucketName,ParameterValue=spam-data-$BUCKET_NAME \
  --capabilities CAPABILITY_IAM
```

Monitor the deployment progress:
```bash
aws cloudformation describe-stacks --stack-name spam-detection-retrain
```

### Step 4: Deploy the Prediction API

Deploy the prediction API using the CloudFormation template:

```bash
aws cloudformation create-stack \
  --stack-name spam-detection-api \
  --template-body file://templates/spam_detection_cloudformation.yaml \
  --parameters ParameterKey=ModelBucketName,ParameterValue=$BUCKET_NAME \
  --capabilities CAPABILITY_IAM
```

### Step 5: Deploy the Frontend

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Update the `.env` file with your API endpoint:
```bash
API_ENDPOINT=https://{your-api-id}.execute-api.{region}.amazonaws.com/prod
```

3. Deploy with Docker:
```bash
docker-compose up -d
```

This will build and run the frontend container, making the application available at http://localhost:3000

## Using the Application

### Spam Detection
1. Navigate to the home page
2. Enter a message in the input field
3. Click "Check Message" to get the prediction result

### Model Retraining
1. Navigate to the "Retrain Model" page
2. Prepare a CSV file with two columns: `message` and `label` (where label is either "spam" or "ham")
3. Upload the CSV file using the upload interface
4. Wait for the retraining process to complete (you'll see a success message with the new model accuracy)

## Monitoring and Troubleshooting

### CloudWatch Logs
Monitor Lambda function execution in CloudWatch:

```bash
# Get the log stream for the retraining function
LOG_GROUP_NAME="/aws/lambda/spam-detection-retrain"
LOG_STREAM=$(aws logs describe-log-streams --log-group-name $LOG_GROUP_NAME --order-by LastEventTime --descending --limit 1 --query 'logStreams[0].logStreamName' --output text)

# View the logs
aws logs get-log-events --log-group-name $LOG_GROUP_NAME --log-stream-name $LOG_STREAM
```

### AWS Console
You can also monitor the system through the AWS Console:
1. **CloudFormation**: View stack status and outputs
2. **Lambda**: Monitor function executions and errors
3. **API Gateway**: Check API request metrics
4. **CloudWatch**: View detailed metrics and logs

## Maintenance and Updates

### Updating the Model
The model automatically updates when new data is uploaded through the retraining interface. The system will:
1. Process the new data
2. Train a new model
3. Save the model artifacts to S3
4. Update the prediction Lambda function

### Updating the Infrastructure
To update the infrastructure:

1. Modify the CloudFormation templates in the `templates/` directory
2. Run the update command:
```bash
aws cloudformation update-stack \
  --stack-name spam-detection-retrain \
  --template-body file://templates/retrain_lambda_template.yaml \
  --parameters ParameterKey=ModelBucketName,ParameterValue=$BUCKET_NAME \
               ParameterKey=UploadBucketName,ParameterValue=spam-data-$BUCKET_NAME \
  --capabilities CAPABILITY_IAM
```

## Cleanup

To remove all resources created by this deployment:

```bash
# Delete the CloudFormation stacks
aws cloudformation delete-stack --stack-name spam-detection-api
aws cloudformation delete-stack --stack-name spam-detection-retrain

# Empty and delete the S3 buckets
aws s3 rm s3://$BUCKET_NAME --recursive
aws s3 rb s3://$BUCKET_NAME --force

aws s3 rm s3://spam-data-$BUCKET_NAME --recursive
aws s3 rb s3://spam-data-$BUCKET_NAME --force

# Stop and remove Docker containers
cd frontend
docker-compose down
```

## Performance Metrics

The system provides the following monitoring metrics through CloudWatch:
- Lambda function invocations
- API Gateway request counts
- Function duration
- Error count and success rate

![CloudWatch Metrics](https://i.imgur.com/eGCvPJG.png)

## Security Considerations

The deployment employs several security best practices:
- Least privilege IAM roles
- CORS configuration for API endpoints
- Input validation in Lambda functions
- Secure S3 bucket configurations

## Reference Screens

### Retraining Interface
![Retraining Interface](https://i.imgur.com/wTpZMnA.png)

### AWS CloudFormation Stacks
![AWS CloudFormation](https://i.imgur.com/FPsWQf5.png)

### Lambda Function Configuration
![Lambda Configuration](https://i.imgur.com/Q8cxF0Q.png)

## Conclusion

This spam detection system demonstrates a complete MLOps pipeline using serverless AWS services. It provides cost-effective, scalable machine learning capabilities with a modern frontend interface. The architecture allows for continuous improvement through model retraining while maintaining high availability and performance.
