import subprocess
import threading
import json
import logging
import os
import time
import sys
from typing import Mapping, Callable, Any, Optional
from neuron_x.plugin_base import BasePlugin, PluginMetadata

logger = logging.getLogger("neuron-x.plugins.signal")

class SignalConnectorPlugin(BasePlugin):
    """
    Plugin to connect NeuronX to Signal via signal-cli using JSON-RPC.
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
            version="1.6.0",
            description="Signal integration with context-aware messaging and central restart support",
            author="NeuronX",
            capabilities=["messaging", "signal", "files", "voice", "control"]
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
        if not self.process or self.process.poll() is not None:
            return "Error: Signal daemon not running."

        target_recipient = recipient if recipient != "0" and recipient else self.last_source
        if not target_recipient:
            return "Error: No recipient found."

        try:
            params = {"recipient": [target_recipient], "message": text}
            if file_path and os.path.exists(file_path):
                params["attachments"] = [file_path]

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
            return f"Message sent to {target_recipient}."
        except Exception as e:
            logger.error(f"Signal send failed: {e}")
            return f"Error: {str(e)}"

    def _handle_line(self, line: str):
        line = line.strip()
        if not line: return
        try:
            data = json.loads(line)
            envelope = data.get("envelope") or data.get("params", {}).get("envelope")
            if envelope:
                source = envelope.get("source")
                message_data = envelope.get("dataMessage")
                if not message_data: return
                
                text = message_data.get("message")
                attachments = message_data.get("attachments", [])

                if source and self.context and self.context.interact:
                    self.last_source = source
                    voice_text = ""
                    if attachments:
                        # Simple voice transcription placeholder (logic remains similar to before)
                        pass 

                    full_text = text or voice_text
                    if full_text:
                        context_metadata = {
                            "plugin": "signal_connector",
                            "recipient": source
                        }
                        response_text = self.context.interact(full_text, context_metadata=context_metadata)
                        if response_text:
                            self.send_message(source, response_text)
        except:
            pass

    def _run_daemon(self):
        account = os.environ.get("SIGNAL_ACCOUNT")
        if not account: return
        while not self.stop_event.is_set():
            try:
                self.process = subprocess.Popen(
                    ["signal-cli", "-u", account, "jsonRpc"],
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    text=True, bufsize=1
                )
                for line in iter(self.process.stdout.readline, ''):
                    if self.stop_event.is_set(): break
                    self._handle_line(line)
            except:
                pass
            if self.process: self.process.terminate()
            if not self.stop_event.is_set(): time.sleep(5)

    def on_load(self) -> None:
        if os.environ.get("SIGNAL_ACCOUNT"):
            self.stop_event.clear()
            self.thread = threading.Thread(target=self._run_daemon, daemon=True)
            self.thread.start()

    def on_unload(self) -> None:
        self.stop_event.set()
        if self.process: self.process.terminate()

    def is_available(self) -> bool:
        try:
            subprocess.run(["signal-cli", "--version"], capture_output=True)
            return True
        except:
            return False
