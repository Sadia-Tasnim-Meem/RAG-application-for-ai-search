import streamlit as st
import requests

st.set_page_config(page_title="Ask your notes", layout = "centered")
st.title("Ask Your Notes")

query = st.text_input("Ask a question:")

if st.button("Ask"):
    with st.spinner("Thinking..."):
        response = requests.post("http://localhost:8000/query", json={"question": query})
        result = response.json()

        st.subheader("🔍 Retrieved Context")
        st.write(result.get("retrieved_context", "No context found."))

        # Safely check if generated_answer exists and is not empty
        generated = result.get("generated_answer")
        if generated:
            st.subheader("💬 Answer")
            st.write(generated)
        else:
            st.warning("No answer generated.")
