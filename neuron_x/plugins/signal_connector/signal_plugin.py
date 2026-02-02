import subprocess
import threading
import json
import logging
import os
from typing import Mapping, Callable, Any, Optional
from neuron_x.plugin_base import BasePlugin, PluginMetadata

logger = logging.getLogger("neuron-x.plugins.signal")

class SignalConnectorPlugin(BasePlugin):
    """
    Plugin to connect NeuronX to Signal via signal-cli.
    Requires signal-cli to be installed and configured on the system.
    """
    
    def __init__(self):
        super().__init__()
        self.process: Optional[subprocess.Popen] = None
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None

    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="signal_connector",
            version="1.0.0",
            description="Signal Messenger integration using signal-cli",
            author="NeuronX",
            capabilities=["messaging", "signal"]
        )

    def get_tools(self) -> Mapping[str, Callable[..., Any]]:
        return {
            "send_signal_message": self.send_message
        }

    def send_message(self, recipient: str, text: str) -> str:
        """
        Sends a message to a Signal recipient.
        
        Args:
            recipient: Phone number (with country code) or group ID
            text: The message content
        """
        account = os.environ.get("SIGNAL_ACCOUNT")
        if not account:
            return "Error: SIGNAL_ACCOUNT not set in environment."

        try:
            cmd = ["signal-cli", "-u", account, "send", "-m", text, recipient]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return f"Message sent to {recipient}."
        except subprocess.CalledProcessError as e:
            logger.error(f"Signal send error: {e.stderr}")
            return f"Failed to send Signal message: {e.stderr}"
        except Exception as e:
            logger.error(f"Unexpected Signal error: {e}")
            return f"Error: {str(e)}"

    def _listen_loop(self):
        """Background thread to receive messages."""
        account = os.environ.get("SIGNAL_ACCOUNT")
        if not account:
            logger.error("SIGNAL_ACCOUNT not set. Listener aborted.")
            return

        logger.info(f"Starting Signal listener for {account}...")
        
        # Start signal-cli in receive mode with JSON output
        cmd = ["signal-cli", "-u", account, "receive", "--output=json"]
        
        try:
            self.process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )

            for line in iter(self.process.stdout.readline, ''):
                if self.stop_event.is_set():
                    break
                
                if not line.strip():
                    continue

                try:
                    data = json.loads(line)
                    # Handle different signal-cli JSON structures
                    envelope = data.get("envelope", {})
                    source = envelope.get("source")
                    message_data = envelope.get("dataMessage", {})
                    text = message_data.get("message")

                    if text and source and self.context and self.context.interact:
                        logger.info(f"Signal message received from {source}")
                        
                        # Process through NeuronX
                        response_text = self.context.interact(text)
                        
                        # Send response back
                        if response_text:
                            self.send_message(source, response_text)
                            
                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    logger.error(f"Error processing Signal line: {e}")

        except Exception as e:
            logger.error(f"Signal listener crashed: {e}")

    def on_load(self) -> None:
        """Start the background listener if SIGNAL_ACCOUNT is set."""
        if os.environ.get("SIGNAL_ACCOUNT"):
            self.stop_event.clear()
            self.thread = threading.Thread(target=self._listen_loop, daemon=True)
            self.thread.start()
            logger.info("Signal listener thread started.")
        else:
            logger.warning("SIGNAL_ACCOUNT not set. Signal listener will not start.")

    def on_unload(self) -> None:
        """Stop the listener and process."""
        self.stop_event.set()
        if self.process:
            self.process.terminate()
        logger.info("Signal connector unloaded.")

    def is_available(self) -> bool:
        """Check if signal-cli is installed."""
        try:
            subprocess.run(["signal-cli", "--version"], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning("signal-cli not found in PATH.")
            return False
