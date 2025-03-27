#!/usr/bin/env node

/**
 * AWS Configuration Checker
 * This script checks AWS credentials and S3 bucket access for the spam detection model
 */

const { S3Client, HeadBucketCommand, ListBucketsCommand } = require('@aws-sdk/client-s3');
const { STSClient, GetCallerIdentityCommand } = require('@aws-sdk/client-sts');

// Load environment variables if .env file exists
try {
  require('dotenv').config();
} catch (e) {
  console.log('dotenv not available, continuing without it');
}

async function checkAwsCredentials() {
  console.log('\n🔍 Checking AWS credentials...');
  
  // Check if AWS environment variables are set
  const accessKeyExists = !!process.env.AWS_ACCESS_KEY_ID;
  const secretKeyExists = !!process.env.AWS_SECRET_ACCESS_KEY;
  const regionExists = !!process.env.AWS_REGION;
  
  console.log(`AWS_ACCESS_KEY_ID: ${accessKeyExists ? '✅ Set' : '❌ Not set'}`);
  console.log(`AWS_SECRET_ACCESS_KEY: ${secretKeyExists ? '✅ Set' : '❌ Not set'}`);
  console.log(`AWS_REGION: ${regionExists ? '✅ Set (' + process.env.AWS_REGION + ')' : '❌ Not set'}`);
  
  if (!accessKeyExists || !secretKeyExists) {
    console.log('\n❌ AWS credentials are not properly configured.');
    console.log('Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables.');
    return false;
  }
  
  // Try to validate credentials with STS
  try {
    const stsClient = new STSClient({
      region: process.env.AWS_REGION || 'us-east-1'
    });
    
    const command = new GetCallerIdentityCommand({});
    const response = await stsClient.send(command);
    
    console.log('\n✅ AWS credentials are valid');
    console.log(`Account ID: ${response.Account}`);
    console.log(`IAM Principal: ${response.Arn}`);
    return true;
  } catch (error) {
    console.log('\n❌ Failed to validate AWS credentials:');
    console.log(error.message);
    return false;
  }
}

async function checkS3Buckets() {
  console.log('\n🔍 Checking S3 buckets...');
  
  const uploadBucket = process.env.UPLOAD_BUCKET_NAME || 'spam-data-mlop-spam-detection-model-1743079054';
  const modelBucket = process.env.MODEL_BUCKET_NAME;
  
  console.log(`Upload bucket name: ${uploadBucket}`);
  console.log(`Model bucket name: ${modelBucket || 'Not set'}`);
  
  try {
    const s3Client = new S3Client({
      region: process.env.AWS_REGION || 'us-east-1'
    });
    
    // List all buckets
    const { Buckets } = await s3Client.send(new ListBucketsCommand({}));
    console.log(`\nFound ${Buckets.length} bucket(s) in your AWS account:`);
    Buckets.forEach(bucket => console.log(`- ${bucket.Name}`));
    
    // Check upload bucket
    try {
      await s3Client.send(new HeadBucketCommand({ Bucket: uploadBucket }));
      console.log(`\n✅ Upload bucket '${uploadBucket}' exists and is accessible`);
    } catch (error) {
      console.log(`\n❌ Cannot access upload bucket '${uploadBucket}':`);
      console.log(error.message);
    }
    
    // Check model bucket if set
    if (modelBucket) {
      try {
        await s3Client.send(new HeadBucketCommand({ Bucket: modelBucket }));
        console.log(`✅ Model bucket '${modelBucket}' exists and is accessible`);
      } catch (error) {
        console.log(`❌ Cannot access model bucket '${modelBucket}':`);
        console.log(error.message);
      }
    }
    
    return true;
  } catch (error) {
    console.log('\n❌ Failed to check S3 buckets:');
    console.log(error.message);
    return false;
  }
}

async function main() {
  console.log('===================================');
  console.log('🔧 AWS Configuration Checker 🔧');
  console.log('===================================');
  
  const credentialsValid = await checkAwsCredentials();
  if (credentialsValid) {
    await checkS3Buckets();
  }
  
  console.log('\n===================================');
}

main().catch(error => {
  console.error('Error running checks:', error);
  process.exit(1);
}); 