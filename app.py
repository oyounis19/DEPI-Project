import os
from dotenv import load_dotenv
import streamlit as st
from utils import setup_pipeline, setup_faiss, needs_context, sys_prompt, sys_prompt_normal, user_template

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

generation_args = {
    "max_new_tokens": 512,
    "return_full_text": False,
    "temperature": 0.0,
    "do_sample": False,
}

@st.cache_resource
def load_pipeline():
    return setup_pipeline()

@st.cache_resource
def load_faiss():
    return setup_faiss()

pipe = load_pipeline()

db = load_faiss()

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

    with st.spinner("Thinking..."):
        try:
            needs_context_bool = needs_context(prompt, api_key)
            # show in a popup
            st.info(f"needs context: {needs_context_bool}")
        except Exception as e:
            needs_context_bool = True
            st.error(f"LLama API call failed with error: {e}")

        messages_for_model = [{"role": "system", "content": sys_prompt if needs_context_bool else sys_prompt_normal}] + [
            {"role": msg["role"], "content": msg["content"]} for msg in st.session_state.messages
        ]

        if needs_context_bool:
            context = ""
            for i, chunk in enumerate(db.similarity_search(prompt, k=2)):
                context += f"**Document {i+1}**\n\n"
                context += chunk.page_content
                context += "\n\n"
            st.info("Context retrieved: " + context)
            messages_for_model[-1]["content"] = user_template.format(context=context[:-4], query=prompt)

        # Call the model with the entire conversation history
        response = pipe(messages_for_model, **generation_args)[0]['generated_text']
        
    with st.chat_message("assistant"):
        st.markdown(response)

    # Append the new response to the conversation history
    st.session_state.messages.append({"role": "assistant", "content": response})