"use client";

import { useState } from "react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { toast } from "sonner";
import { Textarea } from "./ui/textarea";
import { cn } from "@/lib/utils";
import { Loader2 } from "lucide-react";

const SAMPLE_MESSAGES = [
  {
    text: "Hey! Just wanted to confirm our meeting tomorrow at 2 PM.",
    type: "ham",
    label: "Meeting Confirmation"
  },
  {
    text: "CONGRATULATIONS! You've won $1,000,000! Click here to claim your prize now!!!",
    type: "spam",
    label: "Prize Scam"
  },
  {
    text: "Your package will be delivered tomorrow between 9 AM and 11 AM.",
    type: "ham",
    label: "Delivery Notice"
  },
  {
    text: "URGENT: Your account has been compromised. Click here to verify your details immediately!",
    type: "spam",
    label: "Account Scam"
  }
];

type PredictSpamFunction = (formData: FormData) => Promise<{
  prediction: string;
  probability?: number;
}>;

export function SpamPredictionForm({
  predictSpam,
}: {
  predictSpam: PredictSpamFunction;
}) {
  const [message, setMessage] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<{ prediction: string; probability?: number } | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!message.trim()) {
      toast.error("Please enter a message");
      return;
    }

    try {
      setLoading(true);
      setResult(null);

      const formData = new FormData();
      formData.append("message", message);

      const result = await predictSpam(formData);
      setResult(result);

      const isSpam = result.prediction.toLowerCase() === "spam";
      toast(
        <div className={cn(
          "font-medium",
          isSpam ? "text-red-700" : "text-emerald-700"
        )}>
          {result.prediction.toUpperCase()}
          <p className={cn(
            "font-normal text-sm mt-1",
            isSpam ? "text-red-600" : "text-emerald-600"
          )}>
            Confidence Level: {(result.probability || 0 * 100).toFixed(2)}%
          </p>
        </div>,
        {
          className: cn(
            "border-2",
            isSpam 
              ? "!bg-red-50 !border-red-200" 
              : "!bg-emerald-50 !border-emerald-200"
          ),
        }
      );

      setMessage("");
    } catch (error) {
      toast.error("Prediction Error", {
        description:
          error instanceof Error
            ? error.message
            : "An unexpected error occurred",
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <div className="mb-6 space-y-2">
        <p className="text-sm text-gray-600">
          <span className="text-lg font-bold text-emerald-600">Ham:</span> Legitimate messages that are not spam.
        </p>
        <p className="text-sm text-gray-600">
          <span className=" text-red-600 text-lg font-bold">Spam:</span> Unwanted, unsolicited messages, often containing malicious content or advertisements.
        </p>
      </div>
      <form onSubmit={handleSubmit} className="space-y-4">
        <div className="space-y-2">
          <label
            htmlFor="message"
            className="block text-sm font-medium text-gray-700"
          >
            Enter Message
          </label>
          <Textarea
            id="message"
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            placeholder="Type your message here..."
            className="w-full h-48"
            required
            disabled={loading}
          />
        </div>
        <Button type="submit" className="w-full" disabled={loading}>
          {loading ? (
            <>
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              Predicting...
            </>
          ) : (
            "Predict Spam"
          )}
        </Button>
        {result && (
          <div className={cn(
            "mt-4 p-4 rounded-lg",
            result.prediction.toLowerCase() === "spam" ? "bg-red-50" : "bg-emerald-50"
          )}>
            <h3 className="font-medium">Result:</h3>
            <p className={cn(
              "text-lg",
              result.prediction.toLowerCase() === "spam" ? "text-red-700" : "text-emerald-700"
            )}>
              This message is classified as:{" "}
              <span className="font-bold">{result.prediction.toUpperCase()}</span>
            </p>
            {result.probability && (
              <p className={cn(
                "text-sm",
                result.prediction.toLowerCase() === "spam" ? "text-red-600" : "text-emerald-600"
              )}>
                Confidence Level: {(result.probability * 100).toFixed(2)}%
              </p>
            )}
          </div>
        )}
      </form>

      <div className="mt-8">
        <div className="flex items-center justify-between">  
        <h3 className="text-lg font-medium mb-4">Sample Messages</h3>
        <p className="text-md font-serif italic text-emerald-600">Click to use message</p>
        </div>
        <div className="grid grid-cols-1 gap-3">
          {SAMPLE_MESSAGES.map((sample, index) => (
            <button
              key={index}
              type="button"
              onClick={() => setMessage(sample.text)}
              className="p-3 rounded-lg border-2 text-start w-full cursor-pointer hover:shadow-sm"
            >
              <div className="font-medium mb-1 text-emerald-600">{sample.label}</div>
              <div className="text-sm line-clamp-2">{sample.text}</div>
            </button>
          ))}
        </div>
      </div>
    </>
  );
}
