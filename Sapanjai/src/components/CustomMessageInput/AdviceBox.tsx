import { View, Text, StyleSheet } from "react-native";
import FontAwesome from "@expo/vector-icons/FontAwesome";

const sentimentColors = {
  green: { background: "#d4edda", text: "#155724" },
  yellow: { background: "#fff3cd", text: "#856404" },
  orange: { background: "#ffe5b4", text: "#8a4b00" },
  red: { background: "#f8d7da", text: "#721c24" },
  grey: { background: "#e2e3e5", text: "#9fa1a3ff" },
};

// Define the props type
type AdviceBoxProps = {
  sentiment: keyof typeof sentimentColors; // ensures sentiment is one of "green"|"yellow"|...
  analyzing: boolean;
  finalAdvice: string;
};

export default function AdviceBox({ sentiment, analyzing, finalAdvice }: AdviceBoxProps) {
  return (
    <View
      style={[
        styles.adviceContainer,
        { backgroundColor: sentimentColors[sentiment].text },
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
          style={{
            color: sentimentColors[sentiment].text,
            flex: 1,            // take remaining space
            flexWrap: "wrap",   // allow wrapping
          }}
        >
          {analyzing ? "We are analyzing your message..." : finalAdvice}
        </Text>
      </View>
    </View>
  );
}


const styles = StyleSheet.create({
  adviceContainer: { paddingHorizontal: 10, paddingVertical: 6, borderBottomWidth: 1 },
  adviceBox: { flexDirection: "row", alignItems: "center", padding: 8, borderRadius: 8 },
});
