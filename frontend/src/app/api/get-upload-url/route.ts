import { NextRequest, NextResponse } from "next/server";
import { S3Client, PutObjectCommand, HeadBucketCommand } from "@aws-sdk/client-s3";
import { getSignedUrl } from "@aws-sdk/s3-request-presigner";
import crypto from "crypto";
import { addLog } from "../logs/route";

// Initialize S3 client
const s3Client = new S3Client({
  region: process.env.AWS_REGION || "us-east-1",
  credentials: {
    accessKeyId: process.env.AWS_ACCESS_KEY_ID || "",
    secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY || "",
  },
});

const bucketName = process.env.UPLOAD_BUCKET_NAME || "spam-data-mlop-spam-detection-model-1743079054";

// Utility function to validate bucket access
async function validateBucketAccess(bucket: string): Promise<boolean> {
  try {
    console.log(`Validating access to bucket: ${bucket}`);
    await s3Client.send(new HeadBucketCommand({ Bucket: bucket }));
    console.log(`✅ Successfully validated access to bucket: ${bucket}`);
    addLog(`Successfully validated access to S3 bucket: ${bucket}`, 'success');
    return true;
  } catch (error) {
    console.error(`❌ Failed to access bucket ${bucket}:`, error);
    addLog(`Failed to access S3 bucket: ${bucket}`, 'error');
    return false;
  }
}

export async function POST(request: NextRequest) {
  try {
    console.log("Get upload URL request received");
    addLog("Preparing S3 upload URL request", 'info');
    
    // Log configuration info for debugging
    console.log(`Using S3 bucket: ${bucketName}`);
    console.log(`Region: ${process.env.AWS_REGION || "us-east-1"}`);
    console.log(`API Keys exist: ${!!process.env.AWS_ACCESS_KEY_ID && !!process.env.AWS_SECRET_ACCESS_KEY}`);
    
    // Validate bucket access first
    const bucketAccessible = await validateBucketAccess(bucketName);
    if (!bucketAccessible) {
      console.error(`Cannot access S3 bucket: ${bucketName}`);
      addLog(`Cannot access S3 bucket: ${bucketName}`, 'error');
      return NextResponse.json(
        { 
          error: "Failed to access S3 bucket. Please check your AWS credentials and bucket configuration.",
          details: "The application cannot access the S3 bucket for uploads."
        },
        { status: 500 }
      );
    }
    
    const { filename } = await request.json();
    console.log(`Requested upload URL for file: ${filename}`);
    addLog(`Processing file upload request for: ${filename}`, 'info');
    
    if (!filename) {
      console.error("Error: Filename is required");
      addLog("Missing filename in upload request", 'error');
      return NextResponse.json(
        { error: "Filename is required" },
        { status: 400 }
      );
    }

    // Generate a unique key for the file
    const fileExtension = filename.split(".").pop();
    const uniqueId = crypto.randomBytes(16).toString("hex");
    const key = `datasets/${uniqueId}-${filename}`;
    console.log(`Generated S3 key: ${key}`);
    addLog(`Generated S3 upload path: datasets/${uniqueId}-${filename}`, 'info');

    // Create command for S3 PUT operation
    const command = new PutObjectCommand({
      Bucket: bucketName,
      Key: key,
      ContentType: "text/csv",
    });

    // Generate pre-signed URL
    console.log("Generating pre-signed URL...");
    const uploadUrl = await getSignedUrl(s3Client, command, {
      expiresIn: 3600, // URL expires in 1 hour
    });
    console.log("Pre-signed URL generated successfully");
    console.log(`URL will expire in 1 hour`);
    addLog(`S3 upload URL successfully generated`, 'success');

    // Log success (but truncate the URL for security)
    const truncatedUrl = uploadUrl.substring(0, 50) + "...";
    console.log(`Upload URL generated: ${truncatedUrl}`);

    return NextResponse.json({
      uploadUrl,
      key,
      bucket: bucketName,
    });
  } catch (error) {
    console.error("Error generating upload URL:", error);
    const errorMessage = error instanceof Error ? error.message : "Unknown error";
    const errorStack = error instanceof Error ? error.stack : "";
    
    console.error(`Error details: ${errorMessage}`);
    console.error(`Stack trace: ${errorStack}`);
    
    addLog(`Failed to generate S3 upload URL: ${errorMessage}`, 'error');
    
    return NextResponse.json(
      { 
        error: "Failed to generate upload URL", 
        message: errorMessage,
        troubleshooting: [
          "Check that AWS credentials are correctly configured",
          "Verify the S3 bucket exists and is accessible",
          "Ensure your IAM role has appropriate S3 permissions",
          "Check network connectivity to AWS services"
        ]
      },
      { status: 500 }
    );
  }
} 