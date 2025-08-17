import {
  View,
  Text,
  Pressable,
  StyleSheet,
  ActivityIndicator,
} from "react-native";
import { MaterialIcons } from "@expo/vector-icons";

type RewriteProps = {
  text: string;
  onAccept: () => void;
  onDismiss: () => void;
  rewriting: boolean;
};

export default function RewriteBox({
  text,
  onAccept,
  onDismiss,
  rewriting,
}: RewriteProps) {
  return (
    <View style={[styles.shadow, styles.box]}>
      {rewriting ? (
        <View style={styles.loadingRow}>
          <ActivityIndicator size="small" color="#007bff" />
          <Text style={styles.loadingText}>Rewriting...</Text>
        </View>
      ) : (
        <Text style={styles.rewriteText}>{text}</Text>
      )}
      <View style={styles.buttonRow}>
        <Pressable onPress={onAccept} style={styles.iconBtn}>
          <MaterialIcons name="check-circle" size={36} color="#28a745" />
        </Pressable>
        <Pressable onPress={onDismiss} style={styles.iconBtn}>
          <MaterialIcons name="cancel" size={36} color="#dc3545" />
        </Pressable>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  shadow: {
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.13,
    shadowRadius: 4,
    elevation: 3,
  },
  box: {
    backgroundColor: "#f7f7fa",
    padding: 16,
    marginHorizontal: 12,
    marginVertical: 8,
    borderRadius: 14,
  },
  loadingRow: {
    flexDirection: "row",
    alignItems: "center",
    marginBottom: 10,
  },
  loadingText: {
    marginLeft: 10,
    fontSize: 16,
    color: "#007bff",
    fontWeight: "500",
  },
  rewriteText: {
    marginBottom: 12,
    fontSize: 16,
    color: "#333",
    fontWeight: "500",
  },
  buttonRow: {
    flexDirection: "row",
    justifyContent: "flex-end",
    marginTop: 4,
  },
  iconBtn: {
    marginLeft: 12,
  },
});
