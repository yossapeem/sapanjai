import { View, StyleSheet } from "react-native";

const sentimentColors = {
  green: { background: "#d4edda", text: "#155724" },
  yellow: { background: "#fff3cd", text: "#856404" },
  orange: { background: "#ffe5b4", text: "#8a4b00" },
  red: { background: "#f8d7da", text: "#721c24" },
  grey: { background: "#e2e3e5", text: "#9fa1a3ff" },
};

// Define the props type
type SentimentIndicatorProp = {
  sentiment: keyof typeof sentimentColors; 
};
export default function SentimentIndicator({ sentiment }: SentimentIndicatorProp) {
  const colors = {
    red: "#ff4d4f",
    yellow: "#faad14",
    orange: "#FF7518",
    grey: "#dbdbdb",
    green: "#52c41a",
  };
  return <View style={[styles.indicator, { backgroundColor: colors[sentiment] }]} />;
}

const styles = StyleSheet.create({
  indicator: { width: 12, height: 12, borderRadius: 6, marginRight: 8 },
});
