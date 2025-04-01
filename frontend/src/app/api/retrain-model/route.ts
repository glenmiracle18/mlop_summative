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
  const startTime = Date.now();
  console.log(`[DEBUG] Retrain request started at: ${new Date(startTime).toISOString()}`);
  
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

    console.log(`[DEBUG] Lambda invocation details:
    - Function: ${lambdaFunctionName}
    - Region: ${process.env.AWS_REGION || "us-east-1"}
    - Bucket: ${bucketName}
    - Key: ${key}
    - Credentials exist: ${!!process.env.AWS_ACCESS_KEY_ID && !!process.env.AWS_SECRET_ACCESS_KEY}
    - Payload: ${JSON.stringify(payload)}`);
    
    addLog(`Invoking AWS Lambda function: ${lambdaFunctionName}`, 'info');

    // Time the Lambda execution
    const lambdaStartTime = Date.now();
    console.log(`[DEBUG] Lambda invocation started at: ${new Date(lambdaStartTime).toISOString()}`);

    // Invoke Lambda function
    const command = new InvokeCommand({
      FunctionName: lambdaFunctionName,
      Payload: Buffer.from(JSON.stringify(payload)),
      InvocationType: "RequestResponse", // Use "Event" for async invocation
    });

    const response = await lambdaClient.send(command);
    const lambdaEndTime = Date.now();
    const lambdaDuration = (lambdaEndTime - lambdaStartTime) / 1000;
    
    console.log(`[DEBUG] Lambda execution took ${lambdaDuration.toFixed(2)} seconds`);
    console.log(`[DEBUG] Lambda response status code: ${response.StatusCode}`);
    console.log(`[DEBUG] Lambda function error: ${response.FunctionError || 'None'}`);
    
    // Parse Lambda response
    let responsePayload;
    let rawPayloadString = "None";
    try {
      if (response.Payload) {
        rawPayloadString = Buffer.from(response.Payload).toString();
        console.log("[DEBUG] Raw Lambda response:", rawPayloadString);
        responsePayload = JSON.parse(rawPayloadString);
      } else {
        responsePayload = null;
        console.log("[DEBUG] No payload returned from Lambda");
      }
    } catch (parseError) {
      console.error("Failed to parse Lambda response:", parseError);
      console.error("Raw response:", rawPayloadString);
      addLog("Failed to parse Lambda response. Check data format.", 'error');
      
      return NextResponse.json({
        error: "Invalid response from Lambda function",
        details: "The retraining service returned an invalid response. Your data may be in the wrong format.",
        debug: {
          rawResponse: rawPayloadString,
          executionTime: lambdaDuration,
          parseError: parseError instanceof Error ? parseError.message : String(parseError)
        },
        troubleshooting: [
          "Ensure your CSV file has exactly two columns named 'message' and 'label'",
          "Labels should be 'spam' or 'ham' (not v1/v2)",
          "Check that your file is properly formatted (no special characters in headers)",
          "Try uploading the file again after fixing these issues"
        ]
      }, { status: 500 });
    }
    
    if (response.StatusCode !== 200) {
      console.error(`Lambda execution failed with status code: ${response.StatusCode}`);
      addLog(`AWS Lambda execution failed with status code: ${response.StatusCode}`, 'error');
      return NextResponse.json(
        { 
          error: "Lambda execution failed", 
          debug: {
            statusCode: response.StatusCode,
            executionTime: lambdaDuration,
            rawResponse: rawPayloadString
          }
        },
        { status: 500 }
      );
    }
    
    if (response.FunctionError) {
      console.error(`Lambda function error: ${response.FunctionError}`);
      console.error(`Error details: ${JSON.stringify(responsePayload)}`);
      addLog(`AWS Lambda function error: ${response.FunctionError}`, 'error');
      return NextResponse.json(
        { 
          error: `Lambda function error: ${response.FunctionError}`, 
          details: responsePayload,
          debug: {
            functionError: response.FunctionError,
            executionTime: lambdaDuration,
            rawResponse: rawPayloadString
          },
          troubleshooting: [
            "Your data format may not match the expected format",
            "Ensure CSV file has 'message' and 'label' columns (not v1/v2)",
            "Labels should be 'spam' or 'ham'"
          ]
        },
        { status: 500 }
      );
    }
    
    if (!responsePayload) {
      console.error("Empty response payload from Lambda");
      addLog(`Retraining failed: Empty response from Lambda function`, 'error');
      return NextResponse.json(
        { 
          error: "Empty response from Lambda function",
          debug: {
            executionTime: lambdaDuration,
            rawResponse: rawPayloadString
          }
        },
        { status: 500 }
      );
    }
    
    let parsedBody;
    let rawBodyString = "None";
    try {
      if (typeof responsePayload.body === 'string') {
        rawBodyString = responsePayload.body;
        parsedBody = JSON.parse(responsePayload.body);
      } else {
        parsedBody = responsePayload.body;
        rawBodyString = JSON.stringify(responsePayload.body);
      }
      console.log("[DEBUG] Parsed response body:", parsedBody);
    } catch (bodyParseError) {
      console.error("Failed to parse response body:", bodyParseError);
      console.error("Raw body:", responsePayload.body);
      
      return NextResponse.json({
        error: "Failed to process Lambda response",
        details: "The response body could not be parsed correctly.",
        debug: {
          rawBody: typeof responsePayload.body === 'string' ? responsePayload.body : JSON.stringify(responsePayload.body),
          executionTime: lambdaDuration,
          parseError: bodyParseError instanceof Error ? bodyParseError.message : String(bodyParseError)
        },
        raw: responsePayload.body
      }, { status: 500 });
    }
    
    if (responsePayload.statusCode !== 200) {
      const errorMessage = parsedBody?.error || "Retraining process failed";
      
      console.error(`Retraining process failed: ${errorMessage}`);
      addLog(`Model retraining failed: ${errorMessage}`, 'error');
      return NextResponse.json(
        { 
          error: errorMessage,
          debug: {
            statusCode: responsePayload.statusCode,
            executionTime: lambdaDuration,
            body: parsedBody
          },
          troubleshooting: [
            "Check your data format - columns should be 'message' and 'label'",
            "Labels should be 'spam' or 'ham', not v1/v2",
            "Make sure your CSV file doesn't have special characters or formatting issues",
            "Upload a plain text CSV with the correct column headers",
            `Lambda execution took only ${lambdaDuration.toFixed(2)} seconds - this may be too fast for actual retraining`
          ]
        },
        { status: responsePayload.statusCode || 500 }
      );
    }

    // Extract and return retraining results
    const result = parsedBody || {};
    const endTime = Date.now();
    const totalDuration = (endTime - startTime) / 1000;
    
    console.log(`[DEBUG] Total API request took ${totalDuration.toFixed(2)} seconds`);
    console.log(`[DEBUG] Lambda execution took ${lambdaDuration.toFixed(2)} seconds`);
    
    // Add a warning if the execution time is suspiciously fast
    if (lambdaDuration < 5) {
      console.warn(`[WARNING] Lambda execution time (${lambdaDuration.toFixed(2)}s) is suspiciously fast for model retraining!`);
      console.warn(`This may indicate the Lambda is not actually performing full retraining or there's an issue with the data.`);
      addLog(`Warning: Model retraining completed very quickly (${lambdaDuration.toFixed(2)}s). Verify if retraining was thorough.`, 'warning');
    }
    
    console.log(`Retraining completed successfully. Accuracy: ${result.model_accuracy}`);
    addLog(`Model retrained successfully! Accuracy: ${(result.model_accuracy * 100).toFixed(2)}%, execution time: ${lambdaDuration.toFixed(2)}s`, 'success');
    
    return NextResponse.json({
      message: result.message || "Model retrained successfully",
      accuracy: result.model_accuracy || 0,
      endpoint: result.endpoint_name,
      debug: {
        executionTime: lambdaDuration,
        totalRequestTime: totalDuration,
        suspiciouslyFast: lambdaDuration < 5,
        rawLambdaResponse: rawPayloadString,
        parsedBody: parsedBody
      }
    });
    
  } catch (error) {
    console.error("Error triggering model retraining:", error);
    const errorMessage = error instanceof Error ? error.message : "Unknown error";
    const errorStack = error instanceof Error ? error.stack : "";
    
    console.error(`Error details: ${errorMessage}`);
    console.error(`Stack trace: ${errorStack}`);
    addLog(`Model retraining failed: ${errorMessage}`, 'error');
    
    const endTime = Date.now();
    const totalDuration = (endTime - startTime) / 1000;
    
    return NextResponse.json(
      { 
        error: "Failed to trigger model retraining", 
        message: errorMessage,
        debug: {
          totalRequestTime: totalDuration,
          stack: errorStack
        },
        troubleshooting: [
          "Make sure your CSV file has exactly two columns: 'message' and 'label'",
          "The 'label' column should only contain 'spam' or 'ham' values",
          "Ensure your file is a valid CSV format with proper headers",
          "Try uploading a smaller file first to test the functionality"
        ]
      },
      { status: 500 }
    );
  }
} 