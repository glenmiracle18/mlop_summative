"use client";

import { useEffect, useState } from 'react';
import { toast } from 'sonner';

type LogType = 'info' | 'error' | 'success' | 'warning';
type LogEntry = {
  message: string;
  type: LogType;
  timestamp: number;
};

/**
 * Component that listens to server logs and displays them as toasts.
 * This should be added to the layout to ensure server logs are shown to the user.
 */
export function LogsListener() {
  const [eventSource, setEventSource] = useState<EventSource | null>(null);
  
  useEffect(() => {
    // Only run on client
    if (typeof window === 'undefined') return;
    
    // Create an EventSource connection to the server logs endpoint
    const source = new EventSource('/api/logs');
    setEventSource(source);
    
    // Handle incoming log events
    source.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        
        // Handle initial data batch
        if (data.type === 'initial') {
          // Show up to 3 most recent logs when connecting
          const recentLogs = data.logs.slice(-3);
          recentLogs.forEach((log: LogEntry) => {
            displayLog(log);
          });
          return;
        }
        
        // Handle regular log entry
        displayLog(data);
      } catch (error) {
        console.error('Error handling log event:', error);
      }
    };
    
    // Error handling
    source.onerror = (error) => {
      console.error('Error with log event source:', error);
      // Try to reconnect
      source.close();
      setTimeout(() => {
        setEventSource(new EventSource('/api/logs'));
      }, 3000);
    };
    
    // Cleanup on unmount
    return () => {
      source.close();
    };
  }, []);
  
  // Display a log message as a toast notification
  const displayLog = (log: LogEntry) => {
    // Filter out sensitive data or noisy messages to avoid cluttering the UI
    if (shouldSkipLogMessage(log.message)) {
      return;
    }
    
    // Clean up the message
    const cleanMessage = formatLogMessage(log.message);
    
    // Display the appropriate toast based on log type
    switch(log.type) {
      case 'error':
        toast.error(`Server Error`, {
          description: cleanMessage,
          duration: 5000
        });
        break;
      case 'warning':
        toast.warning(`Warning`, {
          description: cleanMessage,
          duration: 3000
        });
        break;
      case 'success':
        toast.success(`Success`, {
          description: cleanMessage,
          duration: 3000
        });
        break;
      case 'info':
        // Only show info logs that are important
        if (isImportantInfoLog(log.message)) {
          toast.info(`Info`, {
            description: cleanMessage,
            duration: 2000
          });
        }
        break;
    }
  };
  
  // Check if this is an important info log that should be shown to the user
  const isImportantInfoLog = (message: string): boolean => {
    const importantPhrases = [
      'retrain',
      'upload',
      'complete',
      'success',
      'progress',
      'model',
      'accuracy',
      's3',
      'bucket',
      'Lambda'
    ];
    
    return importantPhrases.some(phrase => 
      message.toLowerCase().includes(phrase.toLowerCase())
    );
  };
  
  // Check if we should skip this log message (to avoid excessive toasts)
  const shouldSkipLogMessage = (message: string): boolean => {
    const ignorePhrases = [
      'GET',
      'HEAD',
      'OPTIONS',
      'trace',
      'debug',
      'Request',
      'token',
      'credentials',
      'session',
      'middleware',
      'url:',
      'auth',
      'size:',
      'time:',
      'validate',
      'fetching',
      'validating'
    ];
    
    return ignorePhrases.some(phrase => 
      message.toLowerCase().includes(phrase.toLowerCase())
    );
  };
  
  // Format the log message to be more user-friendly
  const formatLogMessage = (message: string): string => {
    // Truncate long messages
    if (message.length > 150) {
      return message.substring(0, 147) + '...';
    }
    
    // Remove timestamps or other noise from the beginning
    if (message.includes(':') && message.length > 10) {
      const parts = message.split(':');
      if (parts[0].length < 20) {
        // If the first part is short, it may be a timestamp or prefix
        return message.substring(parts[0].length + 1).trim();
      }
    }
    
    return message;
  };
  
  // This component doesn't render anything visible
  return null;
} 