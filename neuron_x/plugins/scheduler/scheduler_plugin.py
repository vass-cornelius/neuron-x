import threading
import time
import logging
import os
from datetime import datetime
from typing import Mapping, Callable, Any
from neuron_x.plugin_base import BasePlugin, PluginMetadata

logger = logging.getLogger("neuron-x.plugins.scheduler")

class SchedulerPlugin(BasePlugin):
    """
    Simple scheduler plugin to run tasks at specific times.
    """
    
    def __init__(self):
        super().__init__()
        self.stop_event = threading.Event()
        self.thread: threading.Thread = None
        self.tasks = []

    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="scheduler",
            version="1.0.2",
            description="Runs scheduled tasks and sends output to Signal",
            author="NeuronX",
            capabilities=["scheduling"]
        )

    def on_load(self) -> None:
        self.stop_event.clear()
        
        # Load recipient from environment or fallback
        recipient = os.environ.get("DAILY_BRIEFING_RECIPIENT")
        
        # Schedule the daily news briefing
        # Default time is 08:00 if not specified, but for testing we check a specific variable
        test_time = os.environ.get("DAILY_BRIEFING_TEST_TIME", "08:00")
        
        if recipient:
            self.tasks.append({
                "time": test_time,
                "skill": "daily-news-briefing",
                "last_run": None,
                "recipient": recipient
            })
            logger.info(f"Scheduled '{self.tasks[0]['skill']}' for {test_time} to {recipient}")
        else:
            logger.warning("Scheduler: DAILY_BRIEFING_RECIPIENT not set. No tasks scheduled.")
        
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
        logger.info("Scheduler plugin loaded and worker thread started.")

    def on_unload(self) -> None:
        self.stop_event.set()
        logger.info("Scheduler plugin unloaded.")

    def _worker(self):
        while not self.stop_event.is_set():
            now = datetime.now()
            current_time = now.strftime("%H:%M")
            today = now.date()

            for task in self.tasks:
                if task["time"] == current_time and task["last_run"] != today:
                    logger.info(f"Triggering scheduled task: {task['skill']} at {current_time}")
                    task["last_run"] = today
                    self._execute_skill(task)
            
            time.sleep(10)

    def _execute_skill(self, task):
        if not self.context or not self.context.interact:
            logger.error("Scheduler: Context not available.")
            return

        try:
            skill_name = task["skill"]
            recipient = task["recipient"]
            
            prompt = f"System-Trigger: Führe den Skill '{skill_name}' aus und gib mir NUR die fertige Zusammenfassung zurück."
            logger.info(f"Executing scheduled prompt: {prompt}")
            
            # Get the response from the LLM
            response_text = self.context.interact(prompt)
            
            if response_text:
                # Use the Signal tool to send the message
                # Note: Tools are registered with their tool names. 
                # The SignalConnectorPlugin registers "send_signal_message"
                send_tool = self.context.get_tool("send_signal_message")
                if not send_tool:
                    # Fallback to generic "send_message" if bridge maps it
                    send_tool = self.context.get_tool("send_message")
                
                if send_tool:
                    send_tool(recipient=recipient, text=response_text)
                    logger.info(f"Scheduled briefing sent to {recipient}")
                else:
                    logger.error("Scheduler: No suitable send tool found (checked send_signal_message, send_message).")
        except Exception as e:
            logger.error(f"Failed to execute scheduled skill: {e}")

    def get_tools(self) -> Mapping[str, Callable[..., Any]]:
        return {}
