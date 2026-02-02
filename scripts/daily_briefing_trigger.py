import subprocess
import os

# Configuration
BOT_NUMBER = os.environ.get("SIGNAL_ACCOUNT")
# User-specific information should be in environment variables
USER_NUMBER = os.environ.get("DAILY_BRIEFING_RECIPIENT")

# We send a message to the bot itself to trigger the LLM logic
TRIGGER_MESSAGE = f"COMMAND_TRIGGER: daily-news-briefing for {USER_NUMBER}"

def trigger():
    if not BOT_NUMBER:
        print("Error: SIGNAL_ACCOUNT environment variable not set.")
        return
    
    if not USER_NUMBER:
        print("Error: DAILY_BRIEFING_RECIPIENT environment variable not set.")
        return

    cmd = [
        "signal-cli",
        "-u", BOT_NUMBER,
        "send",
        "-m", TRIGGER_MESSAGE,
        BOT_NUMBER
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"Trigger sent to {BOT_NUMBER} for recipient {USER_NUMBER}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to send trigger: {e.stderr.decode() if e.stderr else 'Unknown error'}")

if __name__ == "__main__":
    trigger()
