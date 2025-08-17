import { PropsWithChildren, useState } from "react";
import { useEffect } from "react";
import { ActivityIndicator } from "react-native";
import { StreamChat } from "stream-chat";
import { Chat, OverlayProvider } from "stream-chat-expo";
import { useAuth } from "./AuthProvider";
import { supabase } from "../lib/supabase";
import { tokenProvider } from "../utils/tokenProvider";

const client = StreamChat.getInstance(process.env.EXPO_PUBLIC_STREAM_API_KEY!);

export default function ChatProvider({ children }: PropsWithChildren) {
  const [isActive, setIsActive] = useState(false);
  const { profile } = useAuth();

  useEffect(() => {
  if (!profile?.id) return;

  const connect = async () => {
    try {
      console.log("Connecting user:", profile.id);
      await client.connectUser(
        {
          id: profile.id,
          name: profile.full_name,
          image: supabase.storage
            .from("avatars")
            .getPublicUrl(profile.avatar_url ?? "").data.publicUrl,
        },
        tokenProvider
      );
      console.log("Connected successfully");
      setIsActive(true);
    } catch (error) {
      console.error("Error connecting user:", error);
    }
  };

  connect();

  return () => {
    console.log("Disconnecting user");
    client.disconnectUser();
    setIsActive(false);
  };
}, [profile?.id]);


  return (
    <OverlayProvider>
      <Chat client={client}>{children}</Chat>
    </OverlayProvider>
  );
}
