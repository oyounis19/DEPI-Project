import os
import gdown

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from peft import PeftModel

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


sys_prompt = """You are a professional medical assistant trained to provide accurate and concise medical information based strictly on the documents provided. Do not include any statements like "based on the documents" or similar phrases in the output. Only give a direct concise answer to the user's question using the information from the documents.
Instructions:
- Base all your responses on the information from the documents provided.
- If the user asks something that doesn't need context, just ignore the context provided, and just respond normally, the user doesn't know anything about the context.
- If the answer is not available in the documents, simply state, "Unfortunately I can't assist with that."
- Keep responses clear and medically accurate.
- Do not be verbose. Keep responses concise.
- Do not exceed 200 tokens per response.
- Format the response in Markdown.
"""

user_template = """context:
{context}
Question:
{query}"""

def download_db():
    # make vector directory
    os.makedirs("vectorstore", exist_ok=True)

    # Direct download links
    index_url = "https://drive.google.com/uc?export=download&id=1K1aO-C280KYGSJp3w8qQyItn5o_GalPM"
    pkl_link = "https://drive.google.com/uc?export=download&id=1VP51Gag4NplhhaWSyIco2lL87mDsefUQ"

    # Download files
    gdown.download(index_url, "vectorstore/index.faiss", quiet=False)
    gdown.download(pkl_link, "vectorstore/index.pkl", quiet=False)

def setup_pipeline():
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=5.0,  # Lower to save more memory, by converting more params to int8
        llm_int8_skip_modules=None  # quantize all modules
    )

    # Load the llm and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct", trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3.5-mini-instruct",
                                                        device_map="auto", 
                                                        quantization_config=bnb_config, 
                                                        trust_remote_code=True)

    # Combine the model with the LORA adapter
    model_with_lora = PeftModel.from_pretrained(base_model, "oyounis/Phi-3.5-instruct-pubmedQA")

    # Create the pipeline
    pipe = pipeline("text-generation",
                    model=model_with_lora,
                    tokenizer=tokenizer,
                    )

    return pipe

def setup_faiss():
    download_db()
    
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    db = FAISS.load_local("vectorstore", embedder, allow_dangerous_deserialization=True)

    return db