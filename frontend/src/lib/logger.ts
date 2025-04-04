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
  
  if (typeof window !== 'undefined') {
    // Client-side: Send log to server
    try {
      fetch('/api/logs/add', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(logEntry),
      }).catch(err => console.error('Failed to send log to server:', err));
    } catch (error) {
      // Silent fail - don't crash if logging fails
    }
  } else {
    // Server-side: Add to buffer directly
    // This is only for server components that import this function
    console.log(`[${type.toUpperCase()}] ${message}`);
  }
  
  return logEntry;
} 