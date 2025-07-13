import { PropsWithChildren, useState } from "react";
import { useEffect } from "react";
import { ActivityIndicator } from "react-native";
import { StreamChat } from "stream-chat";
import { Chat, OverlayProvider } from "stream-chat-expo";
import { useAuth } from "./AuthProvider";

const client = StreamChat.getInstance(process.env.EXPO_PUBLIC_STREAM_API_KEY!);

export default function ChatProvider({ children }: PropsWithChildren) {
  const [isActive, setIsActive] = useState(false);
  const { profile } = useAuth();
  
  useEffect(() => {
    console.log("USE EFFECT: ", profile)
    if (!profile) {
      return;
    }
    const connect = async () => {
      console.log(profile.id);
      await client.connectUser(
        {
          id: profile.id,
          name: profile.full_name,
          image: "https://i.imgur.com/fR9Jz14.png",
        },
        client.devToken(profile.id)
      );
      setIsActive(true);
      //  const channel = client.channel("messaging", "the_park", {
      //    name: "The Park",
      //  }as any);
      //  await channel.watch();
    };

    connect();

    return () => {
      if (isActive) {
        client.disconnectUser();
      }
      setIsActive(false);
    };
  }, [profile?.id]);

  if (!isActive) {
    return <ActivityIndicator />;
  }

  return (
    <OverlayProvider>
      <Chat client={client}>{children}</Chat>
    </OverlayProvider>
  );
}
