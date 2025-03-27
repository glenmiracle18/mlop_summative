import { NextRequest } from "next/server";

// Log buffer to store recent logs
const logBuffer: {message: string, type: string, timestamp: number}[] = [];
const MAX_BUFFER_SIZE = 100;

// Add a log message to the buffer
export function addLog(message: string, type: 'info' | 'error' | 'success' | 'warning' = 'info') {
  const logEntry = {
    message,
    type,
    timestamp: Date.now()
  };
  
  logBuffer.push(logEntry);
  
  // Keep buffer size limited
  if (logBuffer.length > MAX_BUFFER_SIZE) {
    logBuffer.shift();
  }
  
  // Notify all active clients
  notifyClients(logEntry);
}

// Active client connections
const clients = new Set<{
  controller: ReadableStreamDefaultController;
  encoder: TextEncoder;
}>();

// Send a log entry to all connected clients
function notifyClients(logEntry: {message: string, type: string, timestamp: number}) {
  const data = `data: ${JSON.stringify(logEntry)}\n\n`;
  
  clients.forEach(client => {
    try {
      const encodedData = client.encoder.encode(data);
      client.controller.enqueue(encodedData);
    } catch (err) {
      console.error('Error sending log to client:', err);
    }
  });
}

// Server-sent events handler
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

// Enhance the original console methods to also add to our log buffer
if (typeof console !== 'undefined') {
  const originalConsoleLog = console.log;
  const originalConsoleError = console.error;
  const originalConsoleWarn = console.warn;
  const originalConsoleInfo = console.info;
  
  console.log = function(...args: any[]) {
    const message = args.map(arg => 
      typeof arg === 'object' ? JSON.stringify(arg) : String(arg)
    ).join(' ');
    
    addLog(message, 'info');
    originalConsoleLog.apply(console, args);
  };
  
  console.error = function(...args: any[]) {
    const message = args.map(arg => 
      typeof arg === 'object' ? JSON.stringify(arg) : String(arg)
    ).join(' ');
    
    addLog(message, 'error');
    originalConsoleError.apply(console, args);
  };
  
  console.warn = function(...args: any[]) {
    const message = args.map(arg => 
      typeof arg === 'object' ? JSON.stringify(arg) : String(arg)
    ).join(' ');
    
    addLog(message, 'warning');
    originalConsoleWarn.apply(console, args);
  };
  
  console.info = function(...args: any[]) {
    const message = args.map(arg => 
      typeof arg === 'object' ? JSON.stringify(arg) : String(arg)
    ).join(' ');
    
    addLog(message, 'info');
    originalConsoleInfo.apply(console, args);
  };
} 