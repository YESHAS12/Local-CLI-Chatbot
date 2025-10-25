from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch

class ModelLoader:
    def __init__(self, model_name: str = "gpt2", device: int = None, max_new_tokens: int = 128):
        
        self.model_name = model_name
        self.device = self._resolve_device(device)
        self.max_new_tokens = max_new_tokens
        self.pipe = None
        self._load()

    def _resolve_device(self, device):
        
        if device is not None:
            return device
        return 0 if torch.cuda.is_available() else -1

    def _load(self):
        print(f"Loading model '{self.model_name}' (device={self.device})... this may take a moment.")
        self.pipe = pipeline(
            "text-generation",
            model=self.model_name,
            tokenizer=self.model_name,
            device=self.device,  # -1 -> CPU
            truncation=True,
            # the pipeline API will accept generation kwargs at call time
        )
        print("Model loaded.")

    def generate(self, prompt: str, temperature: float = 0.7, top_p: float = 0.9, do_sample: bool = True):
        """
        Returns generated text (string) for the prompt.
        """
        if self.pipe is None:
            raise RuntimeError("Model pipeline not loaded.")
        out = self.pipe(
            prompt,
            max_new_tokens=self.max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            return_full_text=False,  
        )
      
        return out[0]["generated_text"].strip()
