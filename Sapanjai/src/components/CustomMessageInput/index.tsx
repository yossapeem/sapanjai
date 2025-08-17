import { SafeAreaView } from "react-native-safe-area-context";
import { View, TextInput, Pressable, StyleSheet, Text } from "react-native";
import FontAwesome from "@expo/vector-icons/FontAwesome";
import Feather from "@expo/vector-icons/Feather";
import { useState, useEffect } from "react";
import * as ImagePicker from "expo-image-picker";
import AdviceBox from "./AdviceBox";
import SentimentIndicator from "./SentimentIndicator";
import RewriteBox from "./RewriteBox";
import { useSentiment } from "../../hooks/useSentiment";
import { useRewrite } from "../../hooks/useRewrite";
export default function CustomMessageInput({ channel }: any) {
  const [text, setText] = useState("");
  const {
    sentiment,
    finalAdvice,
    analyzing,
    handleChange,
    resetSentiment,
    sentiment_green,
    final_advice_rewrite,
  } = useSentiment();

  const {
    rewrittenText,
    showRewrite,
    rewriting,
    handleRewrite,
    acceptRewrite,
    dismissRewrite,
  } = useRewrite(setText, sentiment_green, final_advice_rewrite);

  const showSuggestRewrite =
    ["red", "orange", "yellow"].includes(sentiment) &&
    !showRewrite &&
    text.trim().length > 0;

  const sendMessage = async () => {
    if (!text.trim()) return;
    await channel.sendMessage({ text });
    setText("");
    resetSentiment();
  };

  const pickAndSendImage = async () => {
    try {
      const permissionResult =
        await ImagePicker.requestMediaLibraryPermissionsAsync();
      if (!permissionResult.granted) {
        alert("Permission denied!");
        return;
      }
      const pickerResult = await ImagePicker.launchImageLibraryAsync({
        quality: 0.7,
      });
      if (pickerResult.canceled || !pickerResult.assets?.[0]?.uri) return;

      const localUri = pickerResult.assets[0].uri;

      // Upload the image to Stream
      const uploadResult = await channel.sendImage(localUri);
      const imageUrl = uploadResult.file; // Use the file property

      if (!imageUrl) {
        alert("Image upload failed.");
        return;
      }

      await channel.sendMessage({
        text: "",
        attachments: [{ type: "image", image_url: imageUrl }],
      });
    } catch (e) {
      console.error("Image send error:", e);
      alert("Failed to send image.");
    }
  };

  useEffect(() => {
    if (text.trim() === "") {
      resetSentiment();
    }
  }, [text]);

  return (
    <SafeAreaView edges={["bottom"]} style={{ backgroundColor: "#fff" }}>
      {text.trim() && !showRewrite && (
        <AdviceBox
          sentiment={sentiment}
          analyzing={analyzing}
          finalAdvice={finalAdvice}
        />
      )}

      {showSuggestRewrite &&
        (rewriting ? (
          <Text
            style={{
              backgroundColor: "#007bff",
              padding: 8,
              borderRadius: 8,
              marginHorizontal: 10,
              marginVertical: 6,
              textAlign: "center",
              color: "#fff",
            }}
          >
            Rewriting...
          </Text>
        ) : (
          <Pressable
            style={{
              backgroundColor: "#007bff",
              padding: 8,
              borderRadius: 8,
              marginHorizontal: 10,
              marginVertical: 6,
              alignItems: "center",
            }}
            onPress={() => handleRewrite(text)}
          >
            <Text style={{ color: "#fff" }}>Suggest Rewrite</Text>
          </Pressable>
        ))}

      {showRewrite && (
        <RewriteBox
          text={rewrittenText}
          onAccept={acceptRewrite}
          onDismiss={dismissRewrite}
          rewriting={rewriting}
        />
      )}

      <View style={styles.inputContainer}>
        <SentimentIndicator sentiment={sentiment} />
        <TextInput
          style={styles.textInput}
          placeholder="Type your message..."
          multiline
          value={text}
          onChangeText={(val) => {
            setText(val);
            handleChange(val);
          }}
        />
        {!text.trim() && (
          <Pressable style={styles.imageBtn} onPress={pickAndSendImage}>
            <Feather name="image" size={20} color="#fff" />
          </Pressable>
        )}
        {text.trim().length > 0 && (
          <Pressable style={styles.sendBtn} onPress={sendMessage}>
            <FontAwesome name="send" size={20} color="#fff" />
          </Pressable>
        )}
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  inputContainer: {
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: "#f1f1f1",
    borderRadius: 24,
    margin: 8,
    paddingHorizontal: 10,
    paddingVertical: 6,
  },
  textInput: {
    flex: 1,
    fontSize: 16,
    paddingVertical: 6,
    paddingHorizontal: 8,
    maxHeight: 120,
  },
  sendBtn: {
    backgroundColor: "#007AFF",
    borderRadius: 20,
    padding: 8,
    marginLeft: 8,
  },
  imageBtn: {
    backgroundColor: "#28a745",
    borderRadius: 20,
    padding: 8,
    marginLeft: 6,
    alignItems: "center",
    justifyContent: "center",
  },
});
