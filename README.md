# Spam Detection MLOps Pipeline
## Author: Bonyu Miracle Glen

![MLOps Architecture](https://i.imgur.com/lmT7w2j.png)

## Overview

This comprehensive project implements a Spam Detection MLOps pipeline on AWS. The system follows MLOps best practices, providing continuous model improvement through a serverless architecture that enables users to classify messages as spam/not spam and retrain the model with new data.

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

## Repository Structure

```
spam-detection-model/
│
├── README.md               # Project documentation
│
├── notebook/               # Jupyter notebooks
│   └── Glen MLOP Summative Pipeline.ipynb  # Contains model evaluation
│
├── src/                    # Source code
│   ├── lambda/             # Lambda functions
│   │   ├── spam_detection_lambda.py  # Prediction lambda
│   │   └── retrain_model_lambda.py   # Retraining lambda
│   └── deploy_api.py       # API deployment script
│
├── data/                   # Data directory
│   └── spam.csv            # Dataset for spam classification
│
├── models/                 # Saved model files
│   ├── spam_detection_model.pkl               # Main model pickle file
│   ├── spam_detection_model_vectorizer.pkl    # Text vectorizer
│   ├── spam_detection_model_label_encoder.pkl # Label encoder
│   └── spam_detection_model_metadata.json     # Model metadata
│
├── templates/              # CloudFormation templates
│   ├── spam_detection_cloudformation.yaml  # Prediction stack
│   └── retrain_lambda_template.yaml        # Retraining stack
│
├── frontend/               # Frontend application (dockerized)
│   ├── Dockerfile          # Frontend container definition
│   └── docker-compose.yml  # Docker compose configuration
│
└── scripts/                # Deployment and utility scripts
    └── prepare_deployment.py  # AWS resource preparation
```

## Key Features
- **Serverless architecture** (pay-per-use)
- **CI/CD integration** through CloudFormation
- **Model retraining capabilities** with user data uploads
- **Real-time spam detection** API
- **Modern React frontend** with intuitive UI
- **Comprehensive monitoring** and logging

## Model Evaluation and Data Insights

Our spam detection model undergoes rigorous evaluation in our Jupyter notebook using multiple metrics:

### Key Metrics
- **Accuracy**: 97.8%
- **Precision**: 94.2% 
- **Recall**: 93.6%
- **F1 Score**: 93.9%

### Data Visualizations

#### 1. Message Length Distribution - Spam vs Ham
![Message Length Distribution](INSERT_IMAGE_URL_HERE)

This visualization shows the distribution of message lengths between spam and legitimate messages. Spam messages tend to be longer on average, with more outliers in the higher ranges, indicating verbose marketing language and calls-to-action.

#### 2. Word Frequency Analysis
![Word Frequency Analysis](INSERT_IMAGE_URL_HERE)

This visualization displays the most common words found in spam vs legitimate messages. Words like "free", "offer", and "win" appear disproportionately in spam messages, while conversational words dominate legitimate messages.

#### 3. Feature Importance
![Feature Importance](INSERT_IMAGE_URL_HERE)

This visualization demonstrates the most important features (words/tokens) that the model uses to make its classifications. The words with higher importance scores have greater influence on the model's decision-making process.

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
pip install -r requirements.txt
```

2. Create the S3 buckets and prepare deployment artifacts:
```bash
# Create a unique bucket name
BUCKET_NAME="spam-detection-model-$(date +%s)"

# Run the preparation script
python scripts/prepare_deployment.py --bucket $BUCKET_NAME
```

This script will:
- Create S3 buckets for model artifacts and data uploads
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

## Reference Screens

### Spam Detection Interface
![Prediction Interface](INSERT_IMAGE_URL_HERE)

### Retraining Interface
![Retraining Interface](https://i.imgur.com/wTpZMnA.png)

## Conclusion

This spam detection MLOps pipeline demonstrates a complete end-to-end machine learning system with automated training, deployment, and monitoring. The system allows users to make predictions and contribute to model improvement through the retraining process, all within a cost-effective serverless architecture.
