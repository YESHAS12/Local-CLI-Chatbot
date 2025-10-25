import argparse
from model_loader import ModelLoader
from chat_memory import SlidingWindowMemory
import textwrap
import sys

DEFAULT_MODEL = "gpt2"

def main():
    parser = argparse.ArgumentParser(description="Local CLI Chatbot using Hugging Face")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Hugging Face model id (default: gpt2)")
    parser.add_argument("--device", type=int, default=None, help="Device id (None auto; -1 CPU; 0 GPU0, ...)")
    parser.add_argument("--max-turns", type=int, default=3, help="Sliding window size (number of recent turns)")
    parser.add_argument("--max-new-tokens", type=int, default=100, help="Max tokens to generate per response")
    args = parser.parse_args()

    loader = ModelLoader(model_name=args.model, device=args.device, max_new_tokens=args.max_new_tokens)
    memory = SlidingWindowMemory(max_turns=args.max_turns)

    system_prompt = (
        "You are a helpful assistant. Keep answers concise and friendly. Use information from the recent conversation only."
    )

    print("Chatbot ready. Type your message and press Enter. Type /exit to quit, /clear to clear memory.\n")
    try:
        while True:
            user_input = input("User: ").strip()
            if not user_input:
                continue
            if user_input.lower() == "/exit":
                print("Exiting chatbot. Goodbye!")
                break
            if user_input.lower() == "/clear":
                memory.clear()
                print("Memory cleared.")
                continue

            prompt = memory.get_prompt(current_user_input=user_input, system_prompt=system_prompt)
            # Generate reply
            generated = loader.generate(prompt, temperature=0.7, top_p=0.9, do_sample=True)
            reply = generated
            cutoff_tokens = ["\nUser:", "\nUser :", "User:", "User :"]
            for ct in cutoff_tokens:
                if ct in reply:
                    reply = reply.split(ct)[0].strip()

           
            if not reply:
                reply = "(no response generated)"

            print("Bot:", reply)
           
            memory.add_turn(user_input, reply)

    except KeyboardInterrupt:
        print("\nInterrupted. Exiting chatbot.")
        sys.exit(0)

if __name__ == "__main__":
    main()
