import subprocess
import threading
import json
import logging
import os
import time
from typing import Mapping, Callable, Any, Optional
from neuron_x.plugin_base import BasePlugin, PluginMetadata

logger = logging.getLogger("neuron-x.plugins.signal")

class SignalConnectorPlugin(BasePlugin):
    """
    Plugin to connect NeuronX to Signal via signal-cli using JSON-RPC.
    This avoids database locking issues by using a single persistent process.
    """
    
    def __init__(self):
        super().__init__()
        self.process: Optional[subprocess.Popen] = None
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._request_id = 1
        self.last_source: Optional[str] = None

    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="signal_connector",
            version="1.3.0",
            description="Signal Messenger integration with attachment support and context-aware messaging",
            author="NeuronX",
            capabilities=["messaging", "signal", "files"]
        )

    def get_tools(self) -> Mapping[str, Callable[..., Any]]:
        return {
            "send_signal_message": self.send_message
        }

    def _get_next_id(self) -> int:
        with self._lock:
            self._request_id += 1
            return self._request_id

    def send_message(self, recipient: str, text: str, file_path: Optional[str] = None) -> str:
        """
        Sends a message to a Signal recipient via the running JSON-RPC process.
        To send to the current conversation, pass "0" as the recipient.
        """
        if not self.process or self.process.poll() is not None:
            logger.error("Signal JSON-RPC process is not running.")
            return "Error: Signal JSON-RPC process is not running."

        # Support for "reply to last"
        target_recipient = recipient
        if recipient == "0" or not recipient:
            target_recipient = self.last_source
            
        if not target_recipient:
            return "Error: No recipient provided and no active conversation found."

        try:
            params = {
                "recipient": [target_recipient],
                "message": text
            }
            
            if file_path and os.path.exists(file_path):
                params["attachments"] = [file_path]
                logger.info(f"Signal JSON-RPC: Attaching file {file_path}")

            request = {
                "jsonrpc": "2.0",
                "method": "send",
                "params": params,
                "id": self._get_next_id()
            }
            
            payload = json.dumps(request) + "\n"
            with self._lock:
                self.process.stdin.write(payload)
                self.process.stdin.flush()
            
            logger.info(f"JSON-RPC: Sent message request to {target_recipient}")
            return f"Message request sent to {target_recipient}."
        except Exception as e:
            logger.error(f"Failed to send JSON-RPC message: {e}")
            return f"Error sending message: {str(e)}"

    def _handle_line(self, line: str):
        line = line.strip()
        if not line:
            return

        try:
            data = json.loads(line)
            
            # Check for incoming message notifications
            method = data.get("method")
            params = data.get("params", {})
            envelope = params.get("envelope", {})
            
            if method == "receive":
                source = envelope.get("source")
                message_data = envelope.get("dataMessage", {})
                text = message_data.get("message")

                if text and source and self.context and self.context.interact:
                    logger.info(f"Signal JSON-RPC: Message received from {source}")
                    
                    # Store last source for tool-based replies
                    self.last_source = source
                    
                    # Process through NeuronX
                    response_text = self.context.interact(text)
                    
                    # Send response back through the same RPC stream
                    if response_text:
                        self.send_message(source, response_text)
            
            # Check for errors in responses
            if "error" in data:
                logger.error(f"Signal JSON-RPC error: {data['error']}")
                
        except json.JSONDecodeError:
            pass
        except Exception as e:
            logger.error(f"Error processing Signal RPC line: {e}")

    def _run_daemon(self):
        """Persistent JSON-RPC process loop."""
        account = os.environ.get("SIGNAL_ACCOUNT")
        if not account:
            logger.error("SIGNAL_ACCOUNT not set. JSON-RPC daemon aborted.")
            return

        while not self.stop_event.is_set():
            cmd = ["signal-cli", "-u", account, "jsonRpc"]
            
            try:
                self.process = subprocess.Popen(
                    cmd, 
                    stdin=subprocess.PIPE, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1
                )

                logger.info(f"Signal JSON-RPC daemon started for {account}")

                for line in iter(self.process.stdout.readline, ''):
                    if self.stop_event.is_set():
                        break
                    self._handle_line(line)

                if self.process:
                    rc = self.process.poll()
                    if rc is not None and rc != 0:
                        err = self.process.stderr.read()
                        logger.error(f"Signal JSON-RPC exited with code {rc}: {err}")

            except Exception as e:
                logger.error(f"Signal JSON-RPC daemon crashed: {e}")
            
            if self.process:
                try:
                    self.process.terminate()
                except:
                    pass
            
            if not self.stop_event.is_set():
                logger.info("Signal JSON-RPC daemon disconnected. Retrying in 5 seconds...")
                time.sleep(5)

    def on_load(self) -> None:
        """Start the JSON-RPC daemon thread."""
        if os.environ.get("SIGNAL_ACCOUNT"):
            self.stop_event.clear()
            self.thread = threading.Thread(target=self._run_daemon, daemon=True)
            self.thread.start()
            logger.info("Signal JSON-RPC daemon thread started.")
        else:
            logger.warning("SIGNAL_ACCOUNT not set. Signal plugin will not start.")

    def on_unload(self) -> None:
        """Stop the daemon and terminate process."""
        self.stop_event.set()
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=2)
            except:
                if self.process:
                    self.process.kill()
        logger.info("Signal JSON-RPC plugin unloaded.")

    def is_available(self) -> bool:
        """Check if signal-cli is available."""
        try:
            subprocess.run(["signal-cli", "--version"], capture_output=True, check=True)
            return True
        except:
            return False
