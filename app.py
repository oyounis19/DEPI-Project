import streamlit as st
from utils import model_with_lora, sys_prompt, pipe

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