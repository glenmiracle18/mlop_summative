import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { predictSpam } from "./actions";
import { SpamPredictionForm } from "@/components/spam-form";
import { FileUploadComponent } from "@/components/file-upload";

export default function PredictionPage() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-center bg-gray-100 p-4 gap-8">
      <Card className="w-full max-w-md">
        <CardHeader>
          <CardTitle className="text-2xl text-center">
            Spam Prediction
          </CardTitle>
        </CardHeader>
        <CardContent>
          <SpamPredictionForm predictSpam={predictSpam} />
        </CardContent>
      </Card>
      
      <FileUploadComponent />
      
      <div className="text-center text-sm text-gray-500 mt-8">
        <p>Upload a CSV file with 'message' and 'label' columns to retrain the model.</p>
        <p>The label should be 'spam' or 'ham' to identify spam and legitimate messages.</p>
      </div>
    </main>
  );
}
