import streamlit as st
# from utils import model_with_lora, sys_prompt, pipe

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

generation_args = {
    "max_new_tokens": 512,
    "return_full_text": False,
    "temperature": 0.0,
    "do_sample": False,
}

st.title("AI Medical Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me medical stuff..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    messages_for_model = [{"role": "system", "content": sys_prompt}] + [
        {"role": msg["role"], "content": msg["content"]} for msg in st.session_state.messages
    ]

    with st.chat_message("assistant"):
        # Call the model with the entire conversation history
        response = pipe(messages_for_model, **generation_args)[0]['generated_text']
        
        st.markdown(response)

        # Append the new response to the conversation history

    st.session_state.messages.append({"role": "assistant", "content": response})