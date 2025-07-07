import { Redirect } from "expo-router";
import { Text, View } from "react-native";

export default function HomeScreen() {
  return <Redirect href={'/(home)/(tabs)' as any} />;
}
