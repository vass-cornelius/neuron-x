import os
import threading
import logging
import asyncio
import sys
from typing import Mapping, Callable, Any, Optional
from neuron_x.plugin_base import BasePlugin, PluginMetadata

logger = logging.getLogger("neuron-x.plugins.discord")

class DiscordConnectorPlugin(BasePlugin):
    """
    Plugin to connect NeuronX to Discord.
    Requires the 'discord.py' package.
    """
    
    def __init__(self):
        super().__init__()
        self.client: Optional[Any] = None
        self.thread: Optional[threading.Thread] = None
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.last_channel_id: Optional[int] = None

    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="discord_connector",
            version="1.3.1",
            description="Discord integration with context-aware messaging and remote restart",
            author="NeuronX",
            capabilities=["messaging", "discord", "files", "control"],
            dependencies=["discord.py"]
        )

    def get_tools(self) -> Mapping[str, Callable[..., Any]]:
        return {
            "send_discord_message": self.send_message_tool,
            "restart_neuron_x": self.restart_tool
        }

    def restart_tool(self) -> str:
        """
        Triggers a graceful restart of the entire NeuronX system.
        Use this when plugins have been updated and need to be reloaded.
        """
        logger.info("Remote restart triggered via Tool.")
        
        if self.context and hasattr(self.context, 'restart') and self.context.restart:
            # Use the bridge's restart function which handles cleanup
            def delayed_restart():
                import time
                time.sleep(2)
                self.context.restart()

            threading.Thread(target=delayed_restart, daemon=True).start()
            return "Restart initiated (graceful). I will be back online in a few seconds."
        else:
            # Fallback to old method if context is missing
            def delayed_restart_fallback():
                import time
                time.sleep(2)
                os.execv(sys.executable, [sys.executable] + sys.argv)

            threading.Thread(target=delayed_restart_fallback, daemon=True).start()
            return "Restart initiated (forceful). I will be back online in a few seconds."

    def send_message_tool(self, channel_id: Any, text: str, file_path: Optional[str] = None) -> str:
        """
        Sends a message (and optionally a file) to a Discord channel.
        To send to the current conversation, pass 0 as the channel_id.
        """
        try:
            cid = int(channel_id)
        except (ValueError, TypeError):
            cid = 0
            
        target_id = cid if cid != 0 else self.last_channel_id
        
        if not target_id:
            return "Error: No channel ID provided and no active conversation found."
            
        if not self.client or not self.loop:
            return "Error: Discord client not running."

        future = asyncio.run_coroutine_threadsafe(
            self._send_async(target_id, text, file_path), self.loop
        )
        try:
            future.result(timeout=15)
            return f"Message sent to channel {target_id}."
        except Exception as e:
            return f"Failed to send Discord message: {str(e)}"

    async def _send_async(self, channel_id: int, text: str, file_path: Optional[str] = None):
        import discord
        channel = self.client.get_channel(channel_id)
        if not channel:
            channel = await self.client.fetch_channel(channel_id)
        
        if channel:
            if file_path and os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    discord_file = discord.File(f)
                    await channel.send(text, file=discord_file)
            else:
                await channel.send(text)
        else:
            raise ValueError(f"Channel {channel_id} not found.")

    def _run_bot(self, token: str):
        """Internal method to run the bot in a thread."""
        import discord
        
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        
        intents = discord.Intents.default()
        intents.message_content = True 
        
        self.client = discord.Client(intents=intents)

        @self.client.event
        async def on_ready():
            logger.info(f"Discord Bot logged in as {self.client.user}")

        @self.client.event
        async def on_message(message):
            if message.author == self.client.user:
                return

            self.last_channel_id = message.channel.id
            
            if message.content.strip().lower() == "!restart":
                await message.channel.send("Initialisiere manuellen Neustart...")
                self.restart_tool()
                return

            if self.context and hasattr(self.context, 'interact'):
                logger.info(f"Discord message from {message.author} in {message.channel}: {message.content}")
                async with message.channel.typing():
                    response_text = self.context.interact(message.content)
                
                if response_text:
                    await message.channel.send(response_text)

        try:
            self.loop.run_until_complete(self.client.start(token))
        except Exception as e:
            logger.error(f"Discord bot error: {e}")
        finally:
            try:
                # Cleanup connection before closing loop
                if self.client and not self.client.is_closed():
                    self.loop.run_until_complete(self.client.close())
            except:
                pass
            self.loop.close()

    def on_load(self) -> None:
        token = os.environ.get("DISCORD_TOKEN")
        if token:
            self.thread = threading.Thread(target=self._run_bot, args=(token,), daemon=True)
            self.thread.start()
            logger.info("Discord connector thread started.")
        else:
            logger.warning("DISCORD_TOKEN not set.")

    def on_unload(self) -> None:
        if self.client and self.loop:
            # We try to close it gracefully
            asyncio.run_coroutine_threadsafe(self.client.close(), self.loop)

    def is_available(self) -> bool:
        try:
            import discord
            return True
        except ImportError:
            return False
