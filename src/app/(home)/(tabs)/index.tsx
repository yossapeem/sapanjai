import { useAuth } from "@/src/providers/AuthProvider";
import { router } from "expo-router";

import { ChannelList } from "stream-chat-expo";

export default function MainTab() {
  const { user } = useAuth();

  return (
    <ChannelList
      filters={{members: {$in: [user.id]}}}
      onSelect={(channel) => router.push(`/channel/${channel.cid}` as any)}
    />
  );
}
