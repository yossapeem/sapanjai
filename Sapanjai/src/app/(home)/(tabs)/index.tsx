import { useAuth } from "@/src/providers/AuthProvider";
import { Link, router, Stack } from "expo-router";
import FontAwesome5 from "@expo/vector-icons/FontAwesome5";
import { ChannelList } from "stream-chat-expo";

export default function MainTab() {
  const { user } = useAuth();

  return (
    <>
      <Stack.Screen
        options={{
          headerRight: () => (
            <Link href={'/(home)/users'} asChild>
              <FontAwesome5
                name="users"
                size={22}
                color="gray"
                style={{ marginHorizontal: 15 }}
              />
            </Link>
          ),
        }}
      />
      <ChannelList
        filters={{ members: { $in: [user.id] } }}
        onSelect={(channel) => router.push(`/channel/${channel.cid}` as any)}
      />
    </>
  );
}
