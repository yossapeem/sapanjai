import { PropsWithChildren, useState } from "react";
import { useEffect } from "react";
import { ActivityIndicator } from "react-native";
import { StreamChat } from "stream-chat";
import { Chat, OverlayProvider } from "stream-chat-expo";

const client = StreamChat.getInstance(process.env.EXPO_PUBLIC_STREAM_API_KEY!);

export default function ChatProvider({ children }: PropsWithChildren) {
  const [isAcitive, setIsActive] = useState(false);

  useEffect(() => {
    const connect = async () => {
      await client.connectUser(
        {
          id: "jlahey",
          name: "Jim Lahey",
          image: "https://i.imgur.com/fR9Jz14.png",
        },
        client.devToken("jlahey")
      );
      setIsActive(true);
      //  const channel = client.channel("messaging", "the_park", {
      //    name: "The Park",
      //  }as any);
      //  await channel.watch();
    };

    connect();

    return () => {
      client.disconnectUser();
      setIsActive(false);
    };
  }, []);

  if (!isAcitive) {
    return <ActivityIndicator />;
  }

  return (
    <OverlayProvider>
      <Chat client={client}>{children}</Chat>
    </OverlayProvider>
  );
}
