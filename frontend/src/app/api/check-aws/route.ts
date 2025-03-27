import { NextResponse } from "next/server";
import { S3Client, ListBucketsCommand, HeadBucketCommand } from "@aws-sdk/client-s3";
import { LambdaClient, ListFunctionsCommand } from "@aws-sdk/client-lambda";
import { addLog } from "../logs/route";

// Define a proper type for the result
interface AwsCheckResult {
  status: 'unknown' | 'success' | 'error';
  aws: {
    credentials: boolean;
    region: string;
  };
  s3: {
    accessible: boolean;
    buckets: string[];
    uploadBucket: string;
    bucketExists: boolean;
  };
  lambda: {
    accessible: boolean;
    functions: string[];
    retrainFunction: string;
    functionExists: boolean;
  };
  errors: string[];
}

export async function GET() {
  const results: AwsCheckResult = {
    status: 'unknown',
    aws: {
      credentials: false,
      region: process.env.AWS_REGION || 'us-east-1',
    },
    s3: {
      accessible: false,
      buckets: [],
      uploadBucket: process.env.UPLOAD_BUCKET_NAME || 'spam-data-mlop-spam-detection-model-1743079054',
      bucketExists: false,
    },
    lambda: {
      accessible: false,
      functions: [],
      retrainFunction: process.env.RETRAIN_LAMBDA_NAME || 'spam-detection-retrain',
      functionExists: false,
    },
    errors: [],
  };

  try {
    console.log("Checking AWS configuration...");
    addLog("Checking AWS configuration", 'info');

    // Check credentials existence
    const hasCredentials = !!process.env.AWS_ACCESS_KEY_ID && !!process.env.AWS_SECRET_ACCESS_KEY;
    results.aws.credentials = hasCredentials;
    
    if (!hasCredentials) {
      results.errors.push("AWS credentials are not configured");
      addLog("AWS credential check failed: Missing API keys", 'error');
    } else {
      // Check S3 access (this also validates credentials)
      try {
        const s3Client = new S3Client({
          region: process.env.AWS_REGION || 'us-east-1',
          credentials: {
            accessKeyId: process.env.AWS_ACCESS_KEY_ID || "",
            secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY || "",
          },
        });
        
        const listBucketsResponse = await s3Client.send(new ListBucketsCommand({}));
        results.s3.accessible = true;
        results.s3.buckets = (listBucketsResponse.Buckets || [])
          .map(b => b.Name || '')
          .filter((name): name is string => !!name);
          
        addLog(`AWS credentials are valid. Found ${results.s3.buckets.length} S3 buckets.`, 'success');
        
        // Check if upload bucket exists
        if (results.s3.uploadBucket) {
          try {
            await s3Client.send(new HeadBucketCommand({ Bucket: results.s3.uploadBucket }));
            results.s3.bucketExists = true;
            addLog(`S3 bucket '${results.s3.uploadBucket}' is accessible`, 'success');
          } catch (bucketError) {
            results.errors.push(`Upload bucket is not accessible: ${results.s3.uploadBucket}`);
            addLog(`S3 bucket check failed: Cannot access '${results.s3.uploadBucket}'`, 'error');
          }
        } else {
          results.errors.push("Upload bucket name is not configured");
          addLog("S3 bucket check failed: No bucket name configured", 'error');
        }
      } catch (s3Error) {
        results.errors.push(`S3 error: ${s3Error instanceof Error ? s3Error.message : String(s3Error)}`);
        addLog("S3 access check failed", 'error');
      }
      
      // Check Lambda access
      try {
        const lambdaClient = new LambdaClient({
          region: process.env.AWS_REGION || 'us-east-1',
          credentials: {
            accessKeyId: process.env.AWS_ACCESS_KEY_ID || "",
            secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY || "",
          },
        });
        
        const listFunctionsResponse = await lambdaClient.send(new ListFunctionsCommand({}));
        results.lambda.accessible = true;
        results.lambda.functions = (listFunctionsResponse.Functions || [])
          .map(f => f.FunctionName || '')
          .filter((name): name is string => !!name);
        
        // Check if retrain function exists
        if (results.lambda.retrainFunction) {
          results.lambda.functionExists = results.lambda.functions.includes(results.lambda.retrainFunction);
          
          if (results.lambda.functionExists) {
            addLog(`Lambda function '${results.lambda.retrainFunction}' is accessible`, 'success');
          } else {
            results.errors.push(`Retrain function does not exist: ${results.lambda.retrainFunction}`);
            addLog(`Lambda function check failed: '${results.lambda.retrainFunction}' not found`, 'error');
          }
        } else {
          results.errors.push("Retrain function name is not configured");
          addLog("Lambda function check failed: No function name configured", 'error');
        }
      } catch (lambdaError) {
        results.errors.push(`Lambda error: ${lambdaError instanceof Error ? lambdaError.message : String(lambdaError)}`);
        addLog("Lambda access check failed", 'error');
      }
    }
    
    // Determine overall status
    if (results.errors.length === 0) {
      results.status = 'success';
      addLog("AWS configuration check completed successfully", 'success');
    } else {
      results.status = 'error';
      addLog(`AWS configuration check failed with ${results.errors.length} errors`, 'error');
    }
  } catch (error) {
    results.status = 'error';
    results.errors.push(`General error: ${error instanceof Error ? error.message : String(error)}`);
    addLog("AWS configuration check encountered an unexpected error", 'error');
  }
  
  return NextResponse.json(results);
} 