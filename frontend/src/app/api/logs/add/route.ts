import { NextRequest, NextResponse } from "next/server";

// Reference to the parent route that already has the log buffer
import { addLogToBuffer } from "../internalUtils";

export async function POST(request: NextRequest) {
  try {
    const logData = await request.json();
    
    if (!logData.message || !logData.type) {
      return NextResponse.json(
        { error: "Missing required fields" },
        { status: 400 }
      );
    }
    
    // Add log to buffer in the parent route
    addLogToBuffer(logData.message, logData.type);
    
    return NextResponse.json({ success: true });
  } catch (error) {
    console.error("Error adding log:", error);
    return NextResponse.json(
      { error: "Failed to process log" },
      { status: 500 }
    );
  }
} 