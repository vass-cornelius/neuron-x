# NEURON-X: Self-Evolving Knowledge Graph AI

NEURON-X is a memory-augmented AI system that uses a relational graph (Cortex) to maintain long-term context and personalized identity. It "perceives" interactions, extracts semantic triples, and "consolidates" them into a persistent graph structure.

## 🚀 Quick Start

### 1. Installation
The project includes an automatic setup script that handles virtual environment creation and dependency installation.

```bash
python3 install.py
```

### 2. Configuration
Create a `.env` file in the root directory (or copy the `.env.example` file) and add your Gemini API key:

```env
GEMINI_API_KEY=your_api_key_here
NEURON_X_LOG_LEVEL=INFO

# Goal Priority Weights (Probabilistic Selection)
GOAL_WEIGHT_CRITICAL=10
GOAL_WEIGHT_HIGH=5
GOAL_WEIGHT_MEDIUM=2
GOAL_WEIGHT_LOW=1
```

### Goal Priorities
The system uses a **weighted probabilistic** selection for its internal thought goals. You can adjust the weights in `.env` to change how often the AI focuses on different types of tasks:

- **CRITICAL** (Default: 10x): Urgent consistency checks, direct contradictions.
- **HIGH** (Default: 5x): Core identity gaps, important user facts.
- **MEDIUM** (Default: 2x): Standard maintenance, elaborating on known facts.
- **LOW** (Default: 1x): Curiosity, wandering attention, pure exploration.

This prevents the AI from getting stuck in a single loop (like "efficiency") by allowing lower-priority but creative goals to occasionally be selected.

### 3. Usage
Once installed and configured, you can start the chat interface using:

```bash
./chat.command
```
*(On macOS, you might need to run `chmod +x chat.command` first.)*

### 4. Background Consciousness (Optional)
To enable the "Self-Reflection" loop where the AI thinks autonomously in the background:

**Start the Core:**
```bash
./start-cognitive-core.command
```
This runs `consciousness_loop.py` as a background process, logging its thoughts to `nohup.out`.

**Stop the Core:**
```bash
./stop-cognitive-core.command
```

**⚠️ CRITICAL:** always use the stop command (or `Ctrl+C` if running manually).
**Why?** The system stores short-term memories in a "Transient Buffer". When you stop the core correctly, it triggers a **Consolidation Event**, where these memories are permanently written to the Graph. Force-quitting will result in **memory loss**.

---

## ⚠️ Important: macOS Multi-line Support

To use multi-line input in the chat interface, the application uses **Alt+Enter** (Option+Enter) for newlines and **Enter** to send. For this to work in a macOS terminal, you must enable "Meta" key support:

### Terminal.app
1. Go to **Settings...** (`Cmd + ,`).
2. Navigate to **Profiles** -> **Keyboard**.
3. Check the box **"Use Option as Meta Key"**.

### iTerm2
1. Go to **Settings...** -> **Profiles** -> **Keys**.
2. Set **"Left Option Key"** to **Esc+**.

---

## 💡 Example Flow

1.  **Launch**: `./chat.command`
2.  **Interact**:
    - `> I am playing a Level 5 Wood Elf Rogue named Kaelen.`
    - `> My sword is called 'Sting' and it glows blue near Orcs.`
3.  **Multi-line**:
    - `> Tell me about the city of Limgrave.`
    - `[Alt+Enter] It is a foggy place with many ruins.`
    - `[Enter]` (Submits both lines).
4.  **Hibernate**: Type `exit` or press `Ctrl+C`.

---

## 🛑 Stopping the Conversation

To ensure your progress is saved and the knowledge graph is updated (Consolidation), always exit the application gracefully:

- Type **`exit`** or **`quit`** and press **Enter**.
- Or press **`Ctrl+C`** (standard terminal interrupt).

The system will display: *"[BRIDGE] Triggering Sleep Cycle for consolidation..."* indicating that your session is being ingrained into the long-term memory.

## 🛠 Project Structure

- `neuron_x.py`: The core cognitive engine and graph management.
- `gemini_interface.py`: The interactive chat UI with memory injection.
- `install.py`: Automated environment setup.
- `chat.command`: Easy-launch script for macOS users.
- `memory_vault/`: Directory where the persistent brain state is saved.
