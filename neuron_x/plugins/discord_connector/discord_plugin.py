import os
import threading
import logging
import asyncio
from typing import Mapping, Callable, Any, Optional, List
from neuron_x.plugin_base import BasePlugin, PluginMetadata

logger = logging.getLogger("neuron-x.plugins.discord")

class DiscordConnectorPlugin(BasePlugin):
    """
    Plugin to connect NeuronX to Discord with auto-splitting for long messages.
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
            version="1.6.0",
            description="Discord integration with message splitting (2000 char limit) and central restart support",
            author="NeuronX",
            capabilities=["messaging", "discord", "files", "control"],
            dependencies=["discord.py"]
        )

    def get_tools(self) -> Mapping[str, Callable[..., Any]]:
        return {
            "send_discord_message": self.send_message_tool
        }

    def _split_text(self, text: str, limit: int = 1900) -> List[str]:
        """Splits text into chunks to respect Discord's character limit."""
        if not text:
            return []
        chunks = []
        while len(text) > limit:
            split_at = text.rfind('\n', 0, limit)
            if split_at == -1:
                split_at = limit
            chunks.append(text[:split_at].strip())
            text = text[split_at:].strip()
        if text:
            chunks.append(text)
        return chunks

    async def _safe_send(self, target: Any, text: str, file_path: Optional[str] = None):
        """Helper to send text chunks and optional files safely."""
        import discord
        try:
            chunks = self._split_text(text)
            
            if file_path and os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    discord_file = discord.File(f)
                    await target.send(chunks[0] if chunks else "", file=discord_file)
                    for chunk in chunks[1:]:
                        if chunk: await target.send(chunk)
            else:
                if not chunks: # Still send something if text is empty but no file
                    return
                for chunk in chunks:
                    if chunk: await target.send(chunk)
        except Exception as e:
            logger.error(f"Discord send error: {e}")

    def restart_tool(self, reason: Optional[str] = None, todos: Optional[str] = None) -> str:
        logger.info(f"Local restart triggered. Reason: {reason}")
        resume_state = {
            "plugin": "discord_connector",
            "channel_id": self.last_channel_id,
            "reason": reason,
            "todos": todos
        }
        if self.context and self.context.restart:
            self.context.restart(resume_state)
            return "Restarting..."
        return "Error: Restart function not available."

    def send_message_tool(self, channel_id: Any, text: str, file_path: Optional[str] = None) -> str:
        try:
            cid = int(channel_id)
        except (ValueError, TypeError):
            cid = 0
            
        target_id = cid if cid != 0 else self.last_channel_id
        if not target_id:
            return "Error: No channel ID found."
        if not self.client or not self.loop:
            return "Error: Discord client not running."

        asyncio.run_coroutine_threadsafe(self._send_to_channel(target_id, text, file_path), self.loop)
        return f"Message queued for channel {target_id}."

    async def _send_to_channel(self, channel_id: int, text: str, file_path: Optional[str] = None):
        import discord
        channel = self.client.get_channel(channel_id)
        if not channel:
            try:
                channel = await self.client.fetch_channel(channel_id)
            except:
                logger.error(f"Channel {channel_id} not found.")
                return
        if channel:
            await self._safe_send(channel, text, file_path)

    def _run_bot(self, token: str):
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
                await message.channel.send("Neustart wird eingeleitet...")
                self.restart_tool(reason="User command !restart")
                return

            if self.context and self.context.interact:
                context_metadata = {
                    "plugin": "discord_connector",
                    "channel_id": message.channel.id
                }
                async with message.channel.typing():
                    response_text = self.context.interact(message.content, context_metadata=context_metadata)
                
                if response_text:
                    await self._safe_send(message.channel, response_text)

        try:
            self.loop.run_until_complete(self.client.start(token))
        except Exception as e:
            logger.error(f"Discord bot error: {e}")
        finally:
            try:
                if self.client and not self.client.is_closed():
                    self.loop.run_until_complete(self.client.close())
            except: pass
            self.loop.close()

    def on_load(self) -> None:
        token = os.environ.get("DISCORD_TOKEN")
        if token:
            self.thread = threading.Thread(target=self._run_bot, args=(token,), daemon=True)
            self.thread.start()

    def on_unload(self) -> None:
        if self.client and self.loop:
            asyncio.run_coroutine_threadsafe(self.client.close(), self.loop)

    def is_available(self) -> bool:
        try:
            import discord
            return True
        except ImportError:
            return False
