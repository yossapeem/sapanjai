import { useState, useRef } from "react";
import { analyzeSentiment } from "../api/replicateApi";

export function useSentiment() {
  const [sentiment, setSentiment] = useState<"green" | "yellow" | "orange" | "red" | "grey">("grey");
  const [finalAdvice, setFinalAdvice] = useState("");
  const [analyzing, setAnalyzing] = useState(false);
  const debounceTimeout = useRef<number | null>(null);
  const latestRequest = useRef<number>(0);

  const handleChange = (text: string) => {
    if (debounceTimeout.current) clearTimeout(debounceTimeout.current);

    setSentiment("grey");
    setFinalAdvice("");

    if (!text.trim()) {
      setAnalyzing(false);
      return;
    }

    setAnalyzing(true);
    const requestId = Date.now();
    latestRequest.current = requestId;

    debounceTimeout.current = window.setTimeout(async () => {
      const startTime = Date.now(); // Start timing
      try {
        const result = await analyzeSentiment(text);
        const endTime = Date.now(); // End timing
        console.log(`Sentiment fetch took ${endTime - startTime} ms`);
        // Only update if this is the latest request
        if (latestRequest.current === requestId) {
          if (result) {
            setSentiment(result.sentiment_level || "grey");
            setFinalAdvice(result.final_advice || "");
          } else {
            setSentiment("grey");
            setFinalAdvice("");
          }
          setAnalyzing(false);
        }
      } catch (err: any) {
        if (err.name === "AbortError") {
          setAnalyzing(false);
          return;
        }
        console.error(err);
        setAnalyzing(false);
      }
    }, 500);
  };

  const sentiment_green = () => setSentiment("green");
  const final_advice_rewrite = () => setFinalAdvice("You are good to go.");
  const resetSentiment = () => {
    setSentiment("grey");
    setFinalAdvice("");
  };

  return { sentiment, finalAdvice, analyzing, handleChange, resetSentiment, sentiment_green, final_advice_rewrite };
}
