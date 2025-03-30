"use client"
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { toast } from "sonner";

export function FileConverter() {
  const [conversionResult, setConversionResult] = useState<string | null>(null);

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    try {
      const fileContent = await file.text();
      const lines = fileContent.split("\n").filter(line => line.trim().length > 0);
      
      // Detect format and convert
      let headers = lines[0].split(",").map(h => h.trim());
      
      // Check if it's already in the right format
      if (headers.length >= 2 && 
         (headers[0].toLowerCase() === "message" && headers[1].toLowerCase() === "label")) {
        toast.info("File is already in the correct format!");
        setConversionResult(null);
        return;
      }
      
      // Check if it's in v1,v2 format (like the example data)
      if (headers.length >= 2 && 
         (headers[0].toLowerCase() === "v1" && headers[1].toLowerCase() === "v2")) {
        // Convert to message,label format
        const converted = ["message,label"];
        
        for (let i = 1; i < lines.length; i++) {
          const cols = lines[i].split(",");
          if (cols.length >= 2) {
            const label = cols[0].trim().toLowerCase();
            // Everything after the first column is the message
            const message = cols.slice(1).join(",").trim().replace(/"/g, '""');
            
            // Add quotes around message to handle commas in the text
            converted.push(`"${message}",${label}`);
          }
        }
        
        setConversionResult(converted.join("\n"));
        toast.success("File converted successfully! Download the fixed version.");
      } else {
        toast.error("Couldn't detect file format. Please make sure your file has v1,v2 columns or similar.");
      }
    } catch (error) {
      console.error("File conversion error:", error);
      toast.error("Failed to convert file. Check the file format.");
    }
  };

  const downloadConvertedFile = () => {
    if (!conversionResult) return;
    
    const blob = new Blob([conversionResult], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "converted_spam_data.csv";
    a.click();
    URL.revokeObjectURL(url);
    
    toast.success("File downloaded successfully!");
  };

  return (
    <Card className="mt-6">
      <CardHeader>
        <CardTitle className="text-lg">Format Converter</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          <div className="text-sm text-gray-600">
            If your file is in the wrong format (like having v1/v2 headers), use this tool to convert it:
          </div>
          
          <div className="grid gap-4">
            <input
              type="file"
              accept=".csv"
              onChange={handleFileChange}
              className="block w-full text-sm text-gray-500
                file:mr-4 file:py-2 file:px-4
                file:rounded-md file:border-0
                file:text-sm file:font-semibold
                file:bg-blue-50 file:text-blue-700
                hover:file:bg-blue-100
              "
            />
            
            {conversionResult && (
              <Button onClick={downloadConvertedFile}>
                Download Converted File
              </Button>
            )}
          </div>
          
          {conversionResult && (
            <div className="mt-4">
              <div className="text-sm font-medium mb-2">Preview:</div>
              <div className="bg-gray-50 p-3 rounded-md text-xs font-mono overflow-auto max-h-40">
                {conversionResult.split("\n").slice(0, 5).map((line, i) => (
                  <div key={i} className={i === 0 ? "font-bold" : ""}>{line}</div>
                ))}
                {conversionResult.split("\n").length > 5 && (
                  <div className="text-gray-400 italic">... {conversionResult.split("\n").length - 5} more rows</div>
                )}
              </div>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
} 