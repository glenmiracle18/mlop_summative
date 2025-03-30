import { NextRequest, NextResponse } from "next/server";
import { clients, logBuffer, addLog as _addLog } from "./internalUtils";

// Server-sent events handler for streaming logs
export async function GET(request: NextRequest) {
  const encoder = new TextEncoder();
  
  // Create a stream
  const stream = new ReadableStream({
    start(controller) {
      const clientConnection = {
        controller,
        encoder
      };
      
      // Register this client
      clients.add(clientConnection);
      
      // Send initial batch of recent logs
      const initialBatch = encoder.encode(`data: ${JSON.stringify({
        type: 'initial',
        logs: logBuffer
      })}\n\n`);
      controller.enqueue(initialBatch);
      
      // Handle client disconnect
      request.signal.addEventListener('abort', () => {
        clients.delete(clientConnection);
      });
    }
  });
  
  // Return the stream as a server-sent event
  return new Response(stream, {
    headers: {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      'Connection': 'keep-alive'
    }
  });
} 