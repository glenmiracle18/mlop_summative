#!/usr/bin/env python3
"""
Script to deploy the spam detection API using CloudFormation.
"""

import boto3
import argparse
import time
import sys
import logging
from botocore.exceptions import ClientError

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def deploy_stack(stack_name, template_file, bucket_name=None):
    """
    Deploy CloudFormation stack for the spam detection API
    
    Args:
        stack_name (str): Name for the CloudFormation stack
        template_file (str): Path to the CloudFormation template file
        bucket_name (str, optional): S3 bucket name containing model files
    
    Returns:
        bool: True if deployment succeeds, False otherwise
    """
    try:
        # Read the CloudFormation template
        with open(template_file, 'r') as file:
            template_body = file.read()
        
        # Create CloudFormation client
        cf = boto3.client('cloudformation')
        
        # Prepare parameters
        parameters = []
        if bucket_name:
            parameters.append({
                'ParameterKey': 'S3BucketName',
                'ParameterValue': bucket_name
            })
        
        # Create or update the CloudFormation stack
        try:
            # Check if the stack exists
            cf.describe_stacks(StackName=stack_name)
            logging.info(f"Updating existing stack: {stack_name}")
            cf.update_stack(
                StackName=stack_name,
                TemplateBody=template_body,
                Parameters=parameters,
                Capabilities=['CAPABILITY_IAM']
            )
            logging.info(f"Stack update initiated. Waiting for stack update to complete...")
            waiter = cf.get_waiter('stack_update_complete')
            waiter.wait(StackName=stack_name)
            logging.info("Stack update completed successfully!")
        except ClientError as e:
            if 'does not exist' in str(e):
                # Stack doesn't exist, create it
                logging.info(f"Creating new stack: {stack_name}")
                cf.create_stack(
                    StackName=stack_name,
                    TemplateBody=template_body,
                    Parameters=parameters,
                    Capabilities=['CAPABILITY_IAM']
                )
                logging.info(f"Stack create initiated. Stack ID: {stack_name}")
                logging.info(f"Waiting for stack create to complete (this may take a few minutes)...")
                waiter = cf.get_waiter('stack_create_complete')
                waiter.wait(StackName=stack_name)
                logging.info("Stack create completed successfully!")
            elif 'No updates are to be performed' in str(e):
                logging.info("No updates are needed to the stack.")
            else:
                logging.error(f"Error creating/updating stack: {e}")
                raise
        
        # Get stack outputs
        response = cf.describe_stacks(StackName=stack_name)
        outputs = response['Stacks'][0]['Outputs']
        
        logger.info("Stack outputs:")
        
        for output in outputs:
            logger.info(f"  {output['OutputKey']}: {output['OutputValue']}")
            
            # If this is the API endpoint, print instructions
            if output['OutputKey'] == 'ApiEndpoint':
                api_endpoint = output['OutputValue']
                logger.info("\nAPI Endpoint is ready for testing!")
                logger.info(f"Test with curl: curl -X POST {api_endpoint} -H 'Content-Type: application/json' -d '{{\"text\": \"Hello, is this spam?\"}}'")
        
        return True
        
    except Exception as e:
        logger.error(f"Error deploying stack: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Deploy spam detection API to AWS')
    parser.add_argument('--stack-name', default='spam-detection-api', help='CloudFormation stack name')
    parser.add_argument('--template', default='spam_detection_cloudformation.yaml', help='CloudFormation template file')
    parser.add_argument('--bucket', default=None, help='S3 bucket name for model files')
    parser.add_argument('--delete-stack', action='store_true', help='Delete the CloudFormation stack')
    args = parser.parse_args()
    
    try:
        cf = boto3.client('cloudformation')
        
        if args.delete_stack:
            logging.info(f"Deleting stack: {args.stack_name}")
            cf.delete_stack(StackName=args.stack_name)
            logging.info(f"Stack deletion initiated. Waiting for stack to be deleted...")
            waiter = cf.get_waiter('stack_delete_complete')
            waiter.wait(StackName=args.stack_name)
            logging.info("Stack deletion completed successfully!")
            return
        
        # Read the CloudFormation template
        with open(args.template, 'r') as file:
            template_body = file.read()
        
        # Prepare parameters
        parameters = []
        if args.bucket:
            parameters.append({
                'ParameterKey': 'S3BucketName',
                'ParameterValue': args.bucket
            })
        
        # Create or update the CloudFormation stack
        try:
            # Check if the stack exists
            cf.describe_stacks(StackName=args.stack_name)
            logging.info(f"Updating existing stack: {args.stack_name}")
            cf.update_stack(
                StackName=args.stack_name,
                TemplateBody=template_body,
                Parameters=parameters,
                Capabilities=['CAPABILITY_IAM']
            )
            logging.info(f"Stack update initiated. Waiting for stack update to complete...")
            waiter = cf.get_waiter('stack_update_complete')
            waiter.wait(StackName=args.stack_name)
            logging.info("Stack update completed successfully!")
        except ClientError as e:
            if 'does not exist' in str(e):
                # Stack doesn't exist, create it
                logging.info(f"Creating new stack: {args.stack_name}")
                cf.create_stack(
                    StackName=args.stack_name,
                    TemplateBody=template_body,
                    Parameters=parameters,
                    Capabilities=['CAPABILITY_IAM']
                )
                logging.info(f"Stack create initiated. Stack ID: {args.stack_name}")
                logging.info(f"Waiting for stack create to complete (this may take a few minutes)...")
                waiter = cf.get_waiter('stack_create_complete')
                waiter.wait(StackName=args.stack_name)
                logging.info("Stack create completed successfully!")
            elif 'No updates are to be performed' in str(e):
                logging.info("No updates are needed to the stack.")
            else:
                logging.error(f"Error creating/updating stack: {e}")
                raise
        
        # Get stack outputs
        response = cf.describe_stacks(StackName=args.stack_name)
        outputs = response['Stacks'][0]['Outputs']
        
        logger.info("Stack outputs:")
        
        for output in outputs:
            logger.info(f"  {output['OutputKey']}: {output['OutputValue']}")
            
            # If this is the API endpoint, print instructions
            if output['OutputKey'] == 'ApiEndpoint':
                api_endpoint = output['OutputValue']
                logger.info("\nAPI Endpoint is ready for testing!")
                logger.info(f"Test with curl: curl -X POST {api_endpoint} -H 'Content-Type: application/json' -d '{{\"text\": \"Hello, is this spam?\"}}'")
        
        return True
        
    except Exception as e:
        logger.error(f"Error deploying stack: {str(e)}")
        return False

if __name__ == '__main__':
    main() 