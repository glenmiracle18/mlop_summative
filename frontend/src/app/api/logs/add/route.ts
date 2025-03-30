import { NextRequest, NextResponse } from "next/server";

// Reference to the parent route that already has the log buffer
import { addLogToBuffer } from "../internalUtils";

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { message, type } = body;
    
    if (!message) {
      return NextResponse.json({ error: "Message is required" }, { status: 400 });
    }
    
    const validTypes = ['info', 'error', 'success', 'warning'];
    const logType = validTypes.includes(type) ? type : 'info';
    
    addLogToBuffer(message, logType as any);
    
    return NextResponse.json({ success: true });
  } catch (error) {
    console.error("Error processing log:", error);
    return NextResponse.json({ error: "Internal server error" }, { status: 500 });
  }
} 