import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import config

class MedicalLLM:
    def __init__(self):
        print("⏳ Loading Tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL_PATH)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        print("⏳ Loading Base Model (4-bit)...")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            config.BASE_MODEL_PATH,
            quantization_config=bnb_config,
            device_map="auto"
        )
        
        print(f"⏳ Loading LoRA Adapter...")
        self.model = PeftModel.from_pretrained(
            self.base_model,
            config.LORA_REPO_ID,
            torch_dtype=torch.float16
        )
        self.model.eval()
        print("✅ Model Ready!")

    def generate(self, user_message: str, context: str = "", max_tokens: int = 300, temperature: float = 0.1):
        # 1. Construct RAG Prompt
        if context:
            system_prompt = (
                "You are a helpful medical assistant. Use the provided context to answer the patient's question.\n"
                "If the answer is not in the context, use your general medical knowledge but mention that it is general advice.\n"
                "Do not invent medical facts.\n\n"
                f"### Context:\n{context}\n\n"
            )
        else:
            system_prompt = "You are a helpful medical assistant. Provide safe, general advice.\n"

        full_prompt = f"{system_prompt}### Patient: {user_message}\n### Doctor:"

        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(config.DEVICE)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=(temperature > 0),
                temperature=temperature if temperature > 0 else 1.0,
                repetition_penalty=1.2,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode only the new tokens
        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return response

llm_engine = MedicalLLM()