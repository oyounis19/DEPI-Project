import streamlit as st
from utils import setup_pipeline, setup_faiss, sys_prompt, user_template

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

    messages_for_model = [{"role": "system", "content": sys_prompt}] + [
        {"role": msg["role"], "content": msg["content"]} for msg in st.session_state.messages
    ]
    context = ""
    for i, chunk in enumerate(db.similarity_search(prompt, k=2)):
        context += f"**Document {i+1}**\n\n"
        context += chunk.page_content
        context += "\n\n"

    messages_for_model[-1]["content"] = user_template.format(context=context[:-4], query=prompt)

    with st.spinner("Thinking..."):
        # Call the model with the entire conversation history
        response = pipe(messages_for_model, **generation_args)[0]['generated_text']
        
    with st.chat_message("assistant"):
        st.markdown(response)

    # Append the new response to the conversation history
    st.session_state.messages.append({"role": "assistant", "content": response})