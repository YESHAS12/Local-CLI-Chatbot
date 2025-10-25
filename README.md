# Local CLI Chatbot (Hugging Face)

A simple local command-line chatbot using a Hugging Face text-generation model.  
Designed for a small, CPU-friendly setup. Maintains short-term memory using a sliding window.

Folder Structure
local_cli_chatbot/
├── `model_loader.py` — loads HF model + pipeline and exposes a `generate()` function.
├── `chat_memory.py` — sliding-window memory of last N turns.
├── `interface.py` — CLI loop that accepts user input, keeps context, and replies.
├── `requirements.txt` — install dependencies.
└── README.md
## Setup (Windows)
1. Clone or download this repo.
2. Create and activate a Python venv:
   - Windows:
     python -m venv .venv
     .venv\Scripts\activate
3. Install dependencies:
   pip install -r requirements.txt

4. To run the project:
    python interface.py --model "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

