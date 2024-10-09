# from groq import Groq

# client = Groq(
#     api_key="gsk_*",
# )
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, Pipeline
from peft import PeftModel

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=5.0, # Lower to save more memory, by converting more params to int8
    llm_int8_skip_modules=None # quantize all modules
)

# Load the llm and tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct", trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3.5-mini-instruct",
                                                device_map="auto", 
                                                quantization_config=bnb_config, 
                                                trust_remote_code=True
)

# Combine the model with the LORA adapter
model_with_lora = PeftModel.from_pretrained(base_model, "oyounis/Phi-3.5-instruct-pubmedQA")

pipe = Pipeline("text-generation",
                model=model_with_lora,
                tokenizer=tokenizer
)

sys_prompt = """You are a professional medical assistant trained to provide accurate and reliable information to patients. 
Instructions:
- Keep responses clear and medically accurate.
- Do not exceed 200 tokens per response.
- Format the response in Markdown.
- Use bullet points for lists, bold for important terms, and italic for emphasis where necessary."""