"use server";

export async function predictSpam(formData: FormData) {
  const message = formData.get("message") as string;

  if (!message) {
    throw new Error("Please enter a message to predict");
  }

  try {
    const endPoint = process.env.PREDICTION_ENDPOINT || "https://5sagsrm9ie.execute-api.us-east-1.amazonaws.com/prod/predict";
    const response = await fetch(
      endPoint,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ text: message }),
      },
    );

    if (!response.ok) {
      throw new Error("Prediction failed");
    }

    const result = await response.json();
    
    const predictionData = JSON.parse(result.body);
    
    return {
      prediction: predictionData.prediction || "Unknown",
      probability: predictionData.confidence || 0
    };
  } catch (error) {
    throw new Error(
      error instanceof Error ? error.message : "An unexpected error occurred",
    );
  }
}
