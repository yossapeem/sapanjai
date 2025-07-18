// components/CustomMessageInput.tsx
import FontAwesome from "@expo/vector-icons/FontAwesome";
import { useEffect, useState } from "react";
import { Pressable, StyleSheet, TextInput, View, Text } from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import type { Channel as ChannelType } from "stream-chat";

export default function CustomMessageInput({
  channel,
}: {
  channel: ChannelType;
}) {
  const [text, setText] = useState("");
  const [sentiment, setSentiment] = useState<
    "green" | "yellow" | "orange" | "red" | "grey"
  >("grey");
  const [pressCount, setPressCount] = useState(0);
  const [finalAdvice, setFinalAdvice] = useState("");
  const [analyzing, setAnalyzing] = useState(false);

  const sentimentColors = {
    green: { background: "#d4edda", text: "#155724" },
    yellow: { background: "#fff3cd", text: "#856404" },
    orange: { background: "#ffe5b4", text: "#8a4b00" },
    red: { background: "#f8d7da", text: "#721c24" },
    grey: { background: "#e2e3e5", text: "#9fa1a3ff" },
  };

  const analyzeSentiment = async (text: string) => {
    console.log("analyzing: ", {text})
    setAnalyzing(true);
    try {
      const response = await fetch("http://192.168.1.120:8000/analyze", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ text: text }),
      });

      console.log("Fetch done, status:", response.status);

      if (!response.ok) {
        throw new Error("Failed to analyze text");
      }
      const result = await response.json();
      setSentiment(result.sentiment_level);
      setFinalAdvice(result.final_advice);
      console.log(
        `sentiment_level: ${result.sentiment_level}, final_advice: ${result.final_advice}`
      );
    } catch (error) {
      console.error("Error analyzing sentiment:", error);
    } finally {
      setAnalyzing(false);
    }
  };

  const handleChange = (value: string) => {
    setText(value);
  };

  const sendMessage = async () => {
    try {
      await channel.sendMessage({ text });
      setText("");
      setSentiment("green");
    } catch (err) {
      console.error("❌ Failed to send message:", err);
    }
    setPressCount(0);
  };

  const handleSend = async () => {
    if (!text.trim()) return;

    setPressCount((prev) => {
      const newCount = prev + 1;

      if (newCount === 1) {
        analyzeSentiment(text);
      } else if (newCount === 2) {
        sendMessage();
      }

      return newCount;
    });
  };

  useEffect(() => {
    if (pressCount !== 0) {
      setPressCount(0);
    }
    if (sentiment !== "grey") {
      setSentiment("grey");
    }
    if (analyzing === true) {
      setAnalyzing(false);
    }if (finalAdvice) {
      setFinalAdvice(""); 
    }
      
  }, [text]);

  return (
    <SafeAreaView edges={["bottom"]} style={{ backgroundColor: "#ffffff" }}>
      {finalAdvice ? (
        <View
          style={[
            styles.adviceContainer,
            { backgroundColor: sentimentColors[sentiment].text },
            { borderBottomColor: sentimentColors[sentiment].background },
          ]}
        >
          <View
            style={[
              styles.adviceBox,
              { backgroundColor: sentimentColors[sentiment].background },
            ]}
          >
            <FontAwesome
              name="lightbulb-o"
              size={18}
              color={sentimentColors[sentiment].text}
              style={{ marginRight: 6 }}
            />
            <Text
              style={[
                styles.adviceText,
                { color: sentimentColors[sentiment].text },
              ]}
            >
              {analyzing ? "Analyzing your message..." : finalAdvice}
            </Text>
          </View>
        </View>
      ) : null}
      <View style={styles.container}>
        <View
          style={[
            styles.indicator,
            sentiment === "red"
              ? { backgroundColor: "#ff4d4f" }
              : sentiment === "yellow"
              ? { backgroundColor: "#faad14" }
              : sentiment === "orange"
              ? { backgroundColor: "#FF7518" }
              : sentiment === "grey"
              ? { backgroundColor: "#dbdbdb" }
              : { backgroundColor: "#52c41a" },
          ]}
        />
        <TextInput
          style={[
            styles.input,
            {
              borderColor:
                !text.trim() || sentiment === "grey"
                  ? "#dbdbdb"
                  : sentiment === "red"
                  ? "#ff4d4f"
                  : sentiment === "yellow"
                  ? "#faad14"
                  : sentiment === "orange"
                  ? "#FF7518"
                  : "#52c41a",
            },
          ]}
          placeholder="Type your message..."
          placeholderTextColor="#9e9e9e"
          value={text}
          onChangeText={handleChange}
        />
        <Pressable
          onPress={handleSend}
          disabled={!text.trim()}
          style={{ padding: 8 }}
        >
          <FontAwesome
            name={text.trim() ? "send" : "send-o"}
            size={24}
            color={pressCount === 1 ? "blue" : "black"}
          />
        </Pressable>
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flexDirection: "row",
    padding: 10,
    alignItems: "center",
    backgroundColor: "#ffffff",
    borderTopWidth: 1,
    borderTopColor: "#dbdbdb",
  },
  indicator: {
    width: 12,
    height: 12,
    borderRadius: 6,
    marginRight: 10,
  },
  input: {
    flex: 1,
    padding: 10,
    backgroundColor: "#ffffff",
    color: "#000000",
    borderWidth: 1,
    borderRadius: 10,
    marginRight: 10,
    fontSize: 16,
  },
  adviceContainer: {
    paddingHorizontal: 10,
    paddingVertical: 6,
    //backgroundColor: "#fff8e1",
    //borderBottomColor: "#faad14",
    borderBottomWidth: 1,
  },

  adviceBox: {
    flexDirection: "row",
    alignItems: "center",
    //backgroundColor: "#fff3cd",
    padding: 8,
    borderRadius: 8,
  },

  adviceText: {
    flex: 1,
    //color: "#856404",
    fontSize: 14,
  },
});
