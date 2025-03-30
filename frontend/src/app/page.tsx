import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { predictSpam } from "./actions";
import { SpamPredictionForm } from "@/components/spam-form";
import { FileUploadComponent } from "@/components/file-upload";
import { Toaster } from "sonner";

export default function PredictionPage() {
  return (
    <main className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 p-6 md:p-8">
      <div className="max-w-6xl mx-auto">
        <header className="mb-10 text-center">
          <h1 className="text-3xl md:text-4xl font-bold text-gray-800 mb-2">Spam Detection System</h1>
          <p className="text-gray-600 max-w-2xl mx-auto">Identify spam messages and help improve the model through retraining</p>
        </header>
        
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Prediction Column */}
          <div className="space-y-6">
            <Card className="bg-white shadow-lg border-0 rounded-xl overflow-hidden">
              <CardHeader className="">
                <CardTitle className="text-2xl font-serif text-black">Message Classification</CardTitle>
              </CardHeader>
              <CardContent className="pt-6">
                <SpamPredictionForm predictSpam={predictSpam} />
              </CardContent>
            </Card>

          </div>

          {/* Retraining Column */}
          <Card className="bg-white shadow-lg border-0 rounded-xl overflow-hidden h-fit">
          <CardHeader className="">
                <CardTitle className="text-2xl text-black font-serif">Retraining Model</CardTitle>
              </CardHeader>
            <CardContent className="pt-6">
              <div className="mb-6">
                <p className="text-gray-700 mb-4">Help improve the spam detection model by uploading labeled examples:</p>
                <FileUploadComponent />
              </div>
              
              <div className="bg-gray-50 p-4 rounded-lg border border-gray-200">
                <h3 className="font-medium text-gray-800 mb-2">Upload Requirements:</h3>
                <ul className="text-sm text-gray-600 space-y-2 list-disc pl-5">
                  <li>CSV file with 'message' and 'label' columns</li>
                  <li>Labels must be 'spam' or 'ham' (legitimate messages)</li>
                  <li>For best results, include at least 50 examples of each type</li>
                </ul>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>

      <footer className="mt-12 text-center text-gray-500 text-sm">
        <p>© 2025 Spam Detection System | Created by Bonyu Miracle Glen</p>
      </footer>

      {/* <Toaster
        position="top-center"
        richColors
        toastOptions={{
          style: {
            border: '1px solid #333',
            borderRadius: '8px',
          },
        }}
      /> */}
    </main>
  );
}
