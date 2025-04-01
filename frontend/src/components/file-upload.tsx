"use client";

import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Loader2, Upload, Check, AlertTriangle } from "lucide-react";
import { cn } from "@/lib/utils";
import { toast } from "sonner";

type FileUploadStatus = "idle" | "uploading" | "success" | "error" | "retraining";

interface RetrainingStatus {
  status: FileUploadStatus;
  message?: string;
  details?: string;
  estimatedTime?: number;
  startTime?: number;
  progress?: number;
}

export function FileUploadComponent() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [uploadStatus, setUploadStatus] = useState<RetrainingStatus>({
    status: "idle"
  });
  const [progressInterval, setProgressInterval] = useState<NodeJS.Timeout | null>(null);
  const [lastToastId, setLastToastId] = useState<string | number | null>(null);
  const [checkingAws, setCheckingAws] = useState(false);

  useEffect(() => {
    return () => {
      if (progressInterval) {
        clearInterval(progressInterval);
      }
    };
  }, [progressInterval]);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0] || null;
    setSelectedFile(file);
    
    // Reset status when a new file is selected
    setUploadStatus({ status: "idle" });
    
    // Clear any existing progress intervals
    if (progressInterval) {
      clearInterval(progressInterval);
      setProgressInterval(null);
    }

    if (file) {
      toast.success(`File selected: ${file.name}`, {
        description: `Size: ${(file.size / 1024).toFixed(2)} KB`
      });
    }
  };

  const validateFile = (file: File): boolean => {
    // Check file type (must be CSV)
    if (!file.name.toLowerCase().endsWith('.csv')) {
      setUploadStatus({
        status: "error",
        message: "Invalid file type",
        details: "Please upload a CSV file with message and label columns."
      });
      return false;
    }

    // Check file size (limit to 10MB)
    if (file.size > 10 * 1024 * 1024) {
      setUploadStatus({
        status: "error",
        message: "File too large",
        details: "Please upload a file smaller than 10MB."
      });
      return false;
    }

    return true;
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      setUploadStatus({
        status: "error",
        message: "No file selected",
        details: "Please select a CSV file to upload."
      });
      toast.error("No file selected", {
        description: "Please select a CSV file to upload."
      });
      return;
    }

    if (!validateFile(selectedFile)) {
      return;
    }

    try {
      console.log("Starting file upload process...");
      toast.info("Starting upload process", {
        description: "Preparing to upload your file to S3"
      });
      setUploadStatus({ status: "uploading" });

      // Create a pre-signed URL for direct S3 upload
      console.log("Requesting pre-signed URL...");
      toast.loading("Requesting upload URL...");
      const getUploadUrlResponse = await fetch('/api/get-upload-url', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ filename: selectedFile.name }),
      });

      if (!getUploadUrlResponse.ok) {
        const errorData = await getUploadUrlResponse.json();
        console.error("Failed to get upload URL:", errorData);
        toast.dismiss();
        toast.error("Failed to get upload URL", {
          description: errorData.message || "Could not generate a pre-signed URL for S3 upload"
        });
        throw new Error(errorData.message || "Failed to get upload URL");
      }

      const { uploadUrl, key, bucket } = await getUploadUrlResponse.json();
      console.log(`Got pre-signed URL for key: ${key} in bucket: ${bucket}`);
      toast.dismiss();
      toast.success("Upload URL generated", {
        description: `Your file will be uploaded as ${key.split('/').pop()}`
      });

      // Upload the file directly to S3
      console.log("Uploading file to S3...");
      const uploadToastId = toast.loading("Uploading file to S3...", {
        description: "0% complete"
      });
      setLastToastId(uploadToastId);
      
      try {
        // For binary files like CSV, we need to use XMLHttpRequest for more reliable binary uploads
        // This avoids some CORS and encoding issues that can happen with fetch for binary uploads
        const uploadPromise = new Promise((resolve, reject) => {
          const xhr = new XMLHttpRequest();
          
          xhr.open('PUT', uploadUrl, true);
          xhr.setRequestHeader('Content-Type', 'text/csv');
          
          xhr.onload = () => {
            if (xhr.status >= 200 && xhr.status < 300) {
              resolve({ ok: true, status: xhr.status });
            } else {
              console.error(`S3 upload failed with status: ${xhr.status}`);
              console.error(`S3 response: ${xhr.responseText}`);
              toast.error(`Upload failed with status: ${xhr.status}`, {
                id: uploadToastId
              });
              reject(new Error(`Failed to upload file to S3: ${xhr.status}`));
            }
          };
          
          xhr.onerror = () => {
            console.error("XHR error during S3 upload");
            toast.error("Network error during upload", {
              id: uploadToastId,
              description: "Please check your connection and try again"
            });
            reject(new Error("Network error occurred during S3 upload"));
          };
          
          xhr.upload.onprogress = (event) => {
            if (event.lengthComputable) {
              const percentComplete = Math.round((event.loaded / event.total) * 100);
              console.log(`Upload progress: ${percentComplete}%`);
              toast.loading(`Uploading: ${percentComplete}%`, {
                id: uploadToastId,
                description: `${(event.loaded / 1024).toFixed(2)} KB of ${(event.total / 1024).toFixed(2)} KB`
              });
            }
          };
          
          xhr.send(selectedFile);
        });
        
        await uploadPromise;
        console.log("File successfully uploaded to S3 using XHR");
        toast.success("File uploaded successfully", {
          id: uploadToastId
        });
      } catch (uploadError) {
        console.error("XHR upload failed, falling back to fetch:", uploadError);
        toast.loading("XHR upload failed, trying alternate method...", {
          id: uploadToastId
        });
        
        // Fall back to fetch API if XHR fails
        try {
      const uploadResponse = await fetch(uploadUrl, {
        method: 'PUT',
        body: selectedFile,
        headers: {
          'Content-Type': 'text/csv',
        },
            mode: 'cors'
      });

      if (!uploadResponse.ok) {
            console.error("S3 upload failed with status:", uploadResponse.status);
            const responseText = await uploadResponse.text();
            console.error("S3 error response:", responseText);
            toast.error(`S3 upload failed: ${uploadResponse.status}`, {
              id: uploadToastId,
              description: responseText || uploadResponse.statusText
            });
            throw new Error(`Failed to upload file to S3: ${uploadResponse.status} ${uploadResponse.statusText}`);
          }
          
          toast.success("File uploaded successfully", {
            id: uploadToastId
          });
        } catch (fetchError) {
          console.error("Both XHR and fetch upload methods failed");
          toast.error("Upload failed", {
            id: uploadToastId,
            description: fetchError instanceof Error ? fetchError.message : String(fetchError)
          });
          throw new Error(`S3 upload failed: ${fetchError instanceof Error ? fetchError.message : String(fetchError)}`);
        }
      }

      console.log("File successfully uploaded to S3");
      setUploadStatus({ status: "success", message: "File uploaded successfully" });

      // Start the retraining process with estimated time
      // Estimate based on file size (larger files take longer to train)
      const estimatedTime = Math.max(60, Math.ceil(selectedFile.size / 1024 / 100)); // Rough estimate: 1 second per 100KB, minimum 60 seconds
      const startTime = Date.now();
      
      setUploadStatus({ 
        status: "retraining", 
        message: "Retraining in progress...", 
        details: `Estimated time: ${Math.ceil(estimatedTime / 60)} minute(s)`,
        estimatedTime,
        startTime,
        progress: 0
      });
      
      const retrainingToastId = toast.loading("Starting model retraining...", {
        description: `Estimated time: ${Math.ceil(estimatedTime / 60)} minute(s)`
      });
      setLastToastId(retrainingToastId);

      // Set up interval to update progress
      const interval = setInterval(() => {
        setUploadStatus((prev) => {
          const elapsedSeconds = (Date.now() - (prev.startTime || startTime)) / 1000;
          const progress = Math.min(Math.round((elapsedSeconds / (prev.estimatedTime || estimatedTime)) * 100), 99);
          
          // Update the toast with progress
          if (progress % 10 === 0 || progress === 99) { // Update the toast every 10% to avoid too many updates
            toast.loading(`Retraining: ${progress}% complete`, {
              id: retrainingToastId,
              description: `Estimated time remaining: ${Math.ceil(((prev.estimatedTime || estimatedTime) - elapsedSeconds) / 60)} minute(s)`
            });
          }
          
          return {
            ...prev,
            progress,
            details: `Estimated time: ${Math.ceil((prev.estimatedTime || estimatedTime) / 60)} minute(s). Progress: ${progress}%`
          };
        });
      }, 1000);
      
      setProgressInterval(interval);

      // Trigger the retraining process
      console.log("Triggering model retraining with key:", key);
      toast.loading("Sending retraining request to AWS Lambda...", {
        id: retrainingToastId
      });
      
      const retrainResponse = await fetch('/api/retrain-model', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ key }),
      });

      // Clear the progress interval
      if (progressInterval) {
        clearInterval(progressInterval);
        setProgressInterval(null);
      }
      
      if (interval) {
        clearInterval(interval);
      }

      if (!retrainResponse.ok) {
        console.error("Retraining request failed with status:", retrainResponse.status);
        const errorData = await retrainResponse.json();
        console.error("Retraining error details:", errorData);
        
        toast.error("Retraining failed", {
          id: retrainingToastId,
          description: errorData.error || errorData.message || "An error occurred during model retraining"
        });
        
        throw new Error(errorData.error || errorData.message || "Retraining process failed");
      }

      const retrainResult = await retrainResponse.json();
      console.log("Retraining completed successfully:", retrainResult);

      toast.success("Model retrained successfully", {
        id: retrainingToastId,
        description: `New model accuracy: ${(retrainResult.accuracy * 100).toFixed(2)}%`
      });

      setUploadStatus({ 
        status: "success", 
        message: "Model retrained successfully", 
        details: `New model accuracy: ${(retrainResult.accuracy * 100).toFixed(2)}%` 
      });

    } catch (error) {
      // Clear any existing progress intervals
      if (progressInterval) {
        clearInterval(progressInterval);
        setProgressInterval(null);
      }
      
      console.error("Error during upload or retraining:", error);
      
      // Ensure we show an error toast
      if (lastToastId) {
        toast.error("Process failed", {
          id: lastToastId,
          description: error instanceof Error ? error.message : "An unexpected error occurred"
        });
      } else {
        toast.error("Upload or retraining failed", {
          description: error instanceof Error ? error.message : "An unexpected error occurred"
        });
      }
      
      setUploadStatus({ 
        status: "error", 
        message: "Upload or retraining failed", 
        details: error instanceof Error ? error.message : "An unexpected error occurred"
      });
    }
  };

  const checkAwsConfig = async () => {
    try {
      setCheckingAws(true);
      toast.loading("Checking AWS configuration...");
      
      const response = await fetch('/api/check-aws');
      const result = await response.json();
      
      if (result.status === 'success') {
        toast.success("AWS configuration is valid", {
          description: `S3 Bucket and Lambda function are accessible`,
          duration: 5000
        });
      } else {
        toast.error("AWS configuration check failed", {
          description: `Found ${result.errors.length} issue(s). Check console for details.`,
          duration: 10000
        });
        
        // Log the detailed errors
        console.error("AWS configuration issues:", result.errors);
      }
    } catch (error) {
      toast.error("AWS configuration check failed", {
        description: error instanceof Error ? error.message : "An unexpected error occurred"
      });
    } finally {
      toast.dismiss();
      setCheckingAws(false);
    }
  };

  return (
    <Card className="w-full max-w-md">
      <CardHeader>
        <CardDescription>
          Upload a new dataset to retrain the spam detection model
        </CardDescription>
      </CardHeader>
      
      <CardContent>
        <div className="space-y-4">
          <div 
            className={cn(
              "border-2 border-dashed rounded-lg p-8 text-center cursor-pointer hover:bg-gray-50 transition-colors",
              selectedFile ? "border-emerald-300 bg-emerald-50" : "border-gray-300"
            )}
            onClick={() => document.getElementById('file-upload')?.click()}
          >
            <input
              id="file-upload"
              type="file"
              accept=".csv"
              className="hidden"
              onChange={handleFileChange}
              disabled={uploadStatus.status === "uploading" || uploadStatus.status === "retraining"}
            />
            
            <Upload className={cn(
              "mx-auto h-10 w-10 mb-3",
              selectedFile ? "text-emerald-500" : "text-gray-400"
            )} />
            
            {selectedFile ? (
              <div>
                <p className="text-sm font-medium">{selectedFile.name}</p>
                <p className="text-xs text-gray-500">
                  {(selectedFile.size / 1024).toFixed(2)} KB
                </p>
              </div>
            ) : (
              <div>
                <p className="text-sm font-medium">Click to upload or drag and drop</p>
                <p className="text-xs text-gray-500">CSV file with message and label columns</p>
              </div>
            )}
          </div>

          {uploadStatus.status !== "idle" && (
            <Alert className={cn(
              "mt-4",
              uploadStatus.status === "uploading" ? "bg-blue-50 border-blue-200" :
              uploadStatus.status === "retraining" ? "bg-yellow-50 border-yellow-200" :
              uploadStatus.status === "success" ? "bg-emerald-50 border-emerald-200" :
              "bg-red-50 border-red-200"
            )}>
              <div className="flex items-center gap-2">
                {uploadStatus.status === "uploading" && <Loader2 className="h-4 w-4 text-blue-500 animate-spin" />}
                {uploadStatus.status === "retraining" && <Loader2 className="h-4 w-4 text-yellow-500 animate-spin" />}
                {uploadStatus.status === "success" && <Check className="h-4 w-4 text-emerald-500" />}
                {uploadStatus.status === "error" && <AlertTriangle className="h-4 w-4 text-red-500" />}
                
                <AlertTitle>
                  {uploadStatus.status === "uploading" ? "Uploading..." :
                   uploadStatus.status === "retraining" ? "Retraining Model..." :
                   uploadStatus.status === "success" ? "Success" :
                   "Error"}
                </AlertTitle>
              </div>
              {uploadStatus.message && (
                <AlertDescription className="mt-2">{uploadStatus.message}</AlertDescription>
              )}
              {uploadStatus.details && (
                <AlertDescription className="mt-1 text-xs">{uploadStatus.details}</AlertDescription>
              )}
              
              {uploadStatus.status === "retraining" && uploadStatus.progress !== undefined && (
                <div className="w-full bg-gray-200 rounded-full h-2 mt-2">
                  <div 
                    className="bg-yellow-500 h-2 rounded-full transition-all duration-300 ease-in-out" 
                    style={{ width: `${uploadStatus.progress}%` }}
                  />
                </div>
              )}
            </Alert>
          )}
          
          {/* AWS Configuration Check button */}
          <div className="mt-4 text-right">
            <Button 
              variant="outline" 
              size="sm" 
              onClick={checkAwsConfig}
              disabled={checkingAws || uploadStatus.status === "uploading" || uploadStatus.status === "retraining"}
              className="text-xs font-normal"
            >
              {checkingAws ? (
                <>
                  <Loader2 className="mr-1 h-3 w-3 animate-spin" />
                  Checking...
                </>
              ) : (
                "Check AWS Configuration"
              )}
            </Button>
          </div>
        </div>
      </CardContent>
      
      <CardFooter>
        <Button 
          onClick={handleUpload} 
          className="w-full" 
          disabled={!selectedFile || uploadStatus.status === "uploading" || uploadStatus.status === "retraining" || checkingAws}
        >
          {uploadStatus.status === "uploading" ? (
            <>
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              Uploading...
            </>
          ) : uploadStatus.status === "retraining" ? (
            <>
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              Retraining...
            </>
          ) : (
            "Upload and Retrain"
          )}
        </Button>
      </CardFooter>
    </Card>
  );
} 