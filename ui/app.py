# Streamlit UI
import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="PDF RAG Bot", page_icon="📄")
st.title("📄 PDF RAG Bot")
st.write("Upload a PDF and ask questions about it.")

# Upload section
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file is not None:
    if st.button("Upload and Index PDF"):
        with st.spinner("Indexing your PDF..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
            response = requests.post(f"{API_URL}/upload", files=files)

            if response.status_code == 200:
                st.success(response.json()["message"])
            else:
                st.error("Upload failed. Try again.")

st.divider()

# Question section
question = st.text_input("Ask a question about your PDF")

if st.button("Ask"):
    if question.strip() == "":
        st.warning("Please type a question first.")
    else:
        with st.spinner("Thinking..."):
            response = requests.post(f"{API_URL}/ask", json={"question": question})

            if response.status_code == 200:
                data = response.json()
                st.subheader("Answer")
                st.write(data["answer"])

                with st.expander("View source chunks"):
                    for i, chunk in enumerate(data["sources"]):
                        st.markdown(f"**Chunk {i+1}:** {chunk[:300]}...")
            else:
                st.error("Something went wrong. Make sure a PDF is uploaded first.")