// src/api/replicate.ts
const REPLICATE_API_TOKEN = process.env.EXPO_PUBLIC_REPLICATE_API_TOKEN;

// Rewrite deployment
const REWRITE_DEPLOYMENT_URL = "https://api.replicate.com/v1/deployments/sapanjai/rewrite-v3/predictions";

// Sentiment deployment
const SENTIMENT_DEPLOYMENT_URL = "https://api.replicate.com/v1/deployments/sapanjai/sentiment-v3/predictions";

if (!REPLICATE_API_TOKEN) {
  console.warn("⚠️ Missing EXPO_PUBLIC_REPLICATE_API_TOKEN in environment variables");
}

// POST to your deployment
async function createDeploymentPrediction(
  deployment: string,
  input: Record<string, any>,
  signal?: AbortSignal
) {
  const res = await fetch(deployment, {
    method: "POST",
    headers: {
      "Authorization": `Token ${REPLICATE_API_TOKEN}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ input }),
    signal,
  });
  const prediction = await res.json();
  if (!prediction.urls?.get) throw new Error("Invalid prediction response: missing get URL");
  return prediction;
}

async function waitForPrediction(getUrl: string, signal?: AbortSignal) {
  let result;
  while (true) {
    const res = await fetch(getUrl, {
      headers: { "Authorization": `Token ${REPLICATE_API_TOKEN}` },
      signal,
    });
    result = await res.json();
    if (["succeeded", "failed"].includes(result.status)) break;
    await new Promise((r) => setTimeout(r, 500));
  }
  if (result.status === "succeeded") return result.output;
  throw new Error("Prediction failed");
}


let currentAbortController: AbortController | null = null;


function delay(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function analyzeSentiment(text: string) {
  if (currentAbortController) currentAbortController.abort();
  currentAbortController = new AbortController();
  const signal = currentAbortController.signal;

  try {
    await new Promise(r => setTimeout(r, 400)); // debounce
    if (signal.aborted) return null;

    const prediction = await createDeploymentPrediction(SENTIMENT_DEPLOYMENT_URL, { text }, signal);
    const output = await waitForPrediction(prediction.urls.get, signal);
    return output ?? null; // ensure null if no output
  } catch (err: any) {
    if (err.name === "AbortError") {
      // Silently handle abort, do not log error
      return null;
    }
    console.error(err); // Only log non-abort errors
    return null;
  } finally {
    if (currentAbortController?.signal === signal) currentAbortController = null;
  }
}

async function rewriteMessage(text: string) {
  if (currentAbortController) currentAbortController.abort();
  currentAbortController = new AbortController();
  const signal = currentAbortController.signal;

  try {
    await new Promise(r => setTimeout(r, 500));
    if (signal.aborted) return null;

    const prediction = await createDeploymentPrediction(REWRITE_DEPLOYMENT_URL, { user_message: text }, signal);
    const output = await waitForPrediction(prediction.urls.get, signal);
    return output ?? null;
  } catch (err: any) {
    if (err.name === "AbortError") {
      // Silently handle abort, do not log error
      return null;
    }
    console.error(err); // Only log non-abort errors
    return null;
  } finally {
    if (currentAbortController?.signal === signal) currentAbortController = null;
  }
}


export {analyzeSentiment, rewriteMessage}
