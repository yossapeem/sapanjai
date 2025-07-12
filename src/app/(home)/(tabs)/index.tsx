import { router } from "expo-router";

import { ChannelList } from "stream-chat-expo";

export default function MainTab() {
  return (
    <ChannelList
      onSelect={(channel) => router.push(`/channel/${channel.cid}` as any)}
    />
  );
}
