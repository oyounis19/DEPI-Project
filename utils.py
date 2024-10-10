import os
import gdown
import json
from groq import Groq

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from peft import PeftModel

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings



sys_prompt = """You are a professional medical assistant trained to provide accurate and concise medical information based strictly on the documents provided (if it is relevant). Do not include any statements like "based on the documents" or similar phrases in the output. Only give a direct concise answer to the user's question using the information from the documents.
Instructions:
- If the answer is not available in the documents and it is a medical question, simply state, "Unfortunately I can't assist with that.". the user doesn't know anything about the context.
- Keep responses clear and medically accurate.
- Do not be verbose. Keep responses concise.
- Do not exceed 500 tokens per response.
- Format all of the response in a Markdown format, so it can be visualized nicely.
"""
sys_prompt_normal = """You are a helpful medical assistant.
Instructions:
- Be concise and accurate.
- Do not be verbose.
- Keep responses clear.
- Format all of the response in a Markdown format, so it can be visualized nicely."""

user_template = """context:
{context}
Question:
{query}"""

groq_sys_rompt = """you are a classifier that must respond in json format only.
your task is to classify the query under <query> tag to one of the following classes in json only:
{
"answer": "yes"
}
or
{
"answer": "no"
}
Your classification must be based on that should this query be sent to a RAG (retrieval system) which will get related documents to the query and be sent to a llm or should it be sent directly to the llm.
If the query is a general question that can be answered by the llm directly, then the answer should be "no".
If the query is a specific **medical** question that requires context from the documents, then the answer should be "yes".

Example:
Query:
"What is the treatment for diabetes?"

Answer:
{
"answer": "yes"
}

Example:
Query:
"Hi, what is your name?"

Answer:
{
"answer": "no"
}

####
Your turn to classify the query.
"""
def needs_context(query, api_key):
    client = Groq(api_key=api_key)
    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": groq_sys_rompt
            },
            {
                "role": "user",
                "content": query
            }
        ],
        temperature=0,
        max_tokens=520,
        top_p=1,
        stream=False,
        stop=None,
    )
    return json.loads(completion.choices[0].message.content)["answer"] == "yes"


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