import { NextResponse } from "next/server";

// This endpoint is for debugging purposes only
// It should be disabled or removed in production
export async function GET() {
  // Create a safe version of the environment variables (masking secrets)
  const safeEnv = {
    // AWS Configuration
    AWS_REGION: process.env.AWS_REGION || "(not set)",
    AWS_ACCESS_KEY_ID: process.env.AWS_ACCESS_KEY_ID ? "✓ (set)" : "✗ (not set)",
    AWS_SECRET_ACCESS_KEY: process.env.AWS_SECRET_ACCESS_KEY ? "✓ (set)" : "✗ (not set)",
    
    // Bucket names
    MODEL_BUCKET_NAME: process.env.MODEL_BUCKET_NAME || "(not set)",
    UPLOAD_BUCKET_NAME: process.env.UPLOAD_BUCKET_NAME || "(not set)",
    
    // Lambda names
    RETRAIN_LAMBDA_NAME: process.env.RETRAIN_LAMBDA_NAME || "(not set)",
    PREDICTION_LAMBDA_NAME: process.env.PREDICTION_LAMBDA_NAME || "(not set)",
    
    // Node environment
    NODE_ENV: process.env.NODE_ENV || "(not set)"
  };
  
  return NextResponse.json({
    environment: safeEnv,
    message: "This endpoint is for debugging only. Do not expose in production."
  });
} 