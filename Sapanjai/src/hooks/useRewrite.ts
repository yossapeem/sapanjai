import { useState, useRef } from "react";
import { rewriteMessage } from "../api/replicateApi";

export function useRewrite(setText: (val: string) => void, setSentimentGreen: () => void, setFinalAdvice: () => void) {
  const [rewrittenText, setRewrittenText] = useState("");
  const [showRewrite, setShowRewrite] = useState(false);
  const [rewriting, setRewriting] = useState(false);

  const debounceTimeout = useRef<number | null>(null);

  const handleRewrite = (text: string) => {
    if (!text.trim()) return;

    // Clear previous debounce
    if (debounceTimeout.current) clearTimeout(debounceTimeout.current);

    debounceTimeout.current = window.setTimeout(async () => {
      setRewriting(true);
      const startTime = Date.now(); // Start timing
      try {
        const output = await rewriteMessage(text);
        const endTime = Date.now(); // End timing
        console.log(`Rewrite fetch took ${endTime - startTime} ms`);
        if (output) {
          setRewrittenText(output);
          setShowRewrite(true);
        }
      } catch (err: any) {
        if (err.name === "AbortError") {
          setRewriting(false);
          return;
        }
        console.error(err);
      } finally {
        setRewriting(false);
      }
    }, 500); // 500ms debounce
  };

  const acceptRewrite = () => {
    setText(rewrittenText);
    setShowRewrite(false);
    setSentimentGreen();
    setFinalAdvice();
  };

  const dismissRewrite = () => setShowRewrite(false);

  return { rewrittenText, showRewrite, rewriting, handleRewrite, acceptRewrite, dismissRewrite };
}
