import { Redirect } from "expo-router";
import { Text, View } from "react-native";

export default function HomeScreen() {
  return <Redirect href={'/(auth)/Login' as any} />;
}
