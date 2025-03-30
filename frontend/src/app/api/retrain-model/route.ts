import { NextRequest, NextResponse } from "next/server";
import { LambdaClient, InvokeCommand } from "@aws-sdk/client-lambda";
import { addLog } from "../logs/internalUtils";

// Initialize Lambda client
const lambdaClient = new LambdaClient({
  region: process.env.AWS_REGION || "us-east-1",
  credentials: {
    accessKeyId: process.env.AWS_ACCESS_KEY_ID || "",
    secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY || "",
  },
});

const lambdaFunctionName = process.env.RETRAIN_LAMBDA_NAME || "spam-detection-retrain";
const bucketName = process.env.UPLOAD_BUCKET_NAME || "spam-detection-data-uploads";

export async function POST(request: NextRequest) {
  try {
    console.log("Retrain model request received");
    addLog("Preparing to retrain spam detection model", 'info');
    
    const { key } = await request.json();
    
    if (!key) {
      console.error("Retrain model error: Missing file key");
      addLog("Retraining failed: Missing file key", 'error');
      return NextResponse.json(
        { error: "File key is required" },
        { status: 400 }
      );
    }

    console.log(`Triggering retraining for file: ${key}`);
    addLog(`Starting retraining process with file: ${key.split('/').pop()}`, 'info');
    
    // Prepare payload for Lambda function
    const payload = {
      bucket: bucketName,
      key: key,
    };

    console.log(`Invoking Lambda function: ${lambdaFunctionName} with payload: ${JSON.stringify(payload)}`);
    addLog(`Invoking AWS Lambda function: ${lambdaFunctionName}`, 'info');

    // Invoke Lambda function
    const command = new InvokeCommand({
      FunctionName: lambdaFunctionName,
      Payload: Buffer.from(JSON.stringify(payload)),
      InvocationType: "RequestResponse", // Use "Event" for async invocation
    });

    const response = await lambdaClient.send(command);
    console.log(`Lambda response received with status code: ${response.StatusCode}`);
    
    // Parse Lambda response
    const responsePayload = response.Payload 
      ? JSON.parse(Buffer.from(response.Payload).toString()) 
      : null;
    
    if (response.StatusCode !== 200) {
      console.error(`Lambda execution failed with status code: ${response.StatusCode}`);
      addLog(`AWS Lambda execution failed with status code: ${response.StatusCode}`, 'error');
      return NextResponse.json(
        { error: "Lambda execution failed" },
        { status: 500 }
      );
    }
    
    if (response.FunctionError) {
      console.error(`Lambda function error: ${response.FunctionError}`);
      console.error(`Error details: ${JSON.stringify(responsePayload)}`);
      addLog(`AWS Lambda function error: ${response.FunctionError}`, 'error');
      return NextResponse.json(
        { error: `Lambda function error: ${response.FunctionError}`, details: responsePayload },
        { status: 500 }
      );
    }
    
    if (!responsePayload) {
      console.error("Empty response payload from Lambda");
      addLog(`Retraining failed: Empty response from Lambda function`, 'error');
      return NextResponse.json(
        { error: "Empty response from Lambda function" },
        { status: 500 }
      );
    }
    
    if (responsePayload.statusCode !== 200) {
      const errorMessage = responsePayload.body 
        ? JSON.parse(responsePayload.body).error 
        : "Retraining process failed";
      
      console.error(`Retraining process failed: ${errorMessage}`);
      addLog(`Model retraining failed: ${errorMessage}`, 'error');
      return NextResponse.json(
        { error: errorMessage },
        { status: responsePayload.statusCode || 500 }
      );
    }

    // Extract and return retraining results
    const result = responsePayload.body ? JSON.parse(responsePayload.body) : {};
    
    console.log(`Retraining completed successfully. Accuracy: ${result.model_accuracy}`);
    addLog(`Model retrained successfully! Accuracy: ${(result.model_accuracy * 100).toFixed(2)}%`, 'success');
    return NextResponse.json({
      message: result.message || "Model retrained successfully",
      accuracy: result.model_accuracy || 0,
      endpoint: result.endpoint_name,
    });
    
  } catch (error) {
    console.error("Error triggering model retraining:", error);
    const errorMessage = error instanceof Error ? error.message : "Unknown error";
    const errorStack = error instanceof Error ? error.stack : "";
    
    console.error(`Error details: ${errorMessage}`);
    console.error(`Stack trace: ${errorStack}`);
    addLog(`Model retraining failed: ${errorMessage}`, 'error');
    
    return NextResponse.json(
      { error: "Failed to trigger model retraining", message: errorMessage },
      { status: 500 }
    );
  }
} 