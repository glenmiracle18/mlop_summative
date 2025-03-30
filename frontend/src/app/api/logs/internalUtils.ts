// Log buffer to store recent logs
export const logBuffer: {message: string, type: string, timestamp: number}[] = [];
const MAX_BUFFER_SIZE = 100;

// Active client connections
export const clients = new Set<{
  controller: ReadableStreamDefaultController;
  encoder: TextEncoder;
}>();

// Add a log message to the buffer
export function addLogToBuffer(message: string, type: 'info' | 'error' | 'success' | 'warning' = 'info') {
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
  
  return logEntry;
}

// For backward compatibility with existing code (IMPORTANT)
export const addLog = addLogToBuffer;

// Send a log entry to all connected clients
export function notifyClients(logEntry: {message: string, type: string, timestamp: number}) {
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

// Set up console method overrides in a safe way
if (typeof console !== 'undefined' && typeof window === 'undefined') {
  try {
    const originalConsoleLog = console.log;
    const originalConsoleError = console.error;
    const originalConsoleWarn = console.warn;
    const originalConsoleInfo = console.info;
    
    console.log = function(...args: any[]) {
      const message = args.map(arg => 
        typeof arg === 'object' ? JSON.stringify(arg) : String(arg)
      ).join(' ');
      
      addLogToBuffer(message, 'info');
      originalConsoleLog.apply(console, args);
    };
    
    console.error = function(...args: any[]) {
      const message = args.map(arg => 
        typeof arg === 'object' ? JSON.stringify(arg) : String(arg)
      ).join(' ');
      
      addLogToBuffer(message, 'error');
      originalConsoleError.apply(console, args);
    };
    
    console.warn = function(...args: any[]) {
      const message = args.map(arg => 
        typeof arg === 'object' ? JSON.stringify(arg) : String(arg)
      ).join(' ');
      
      addLogToBuffer(message, 'warning');
      originalConsoleWarn.apply(console, args);
    };
    
    console.info = function(...args: any[]) {
      const message = args.map(arg => 
        typeof arg === 'object' ? JSON.stringify(arg) : String(arg)
      ).join(' ');
      
      addLogToBuffer(message, 'info');
      originalConsoleInfo.apply(console, args);
    };
  } catch (err) {
    // Silently fail if we can't override console methods
  }
} 