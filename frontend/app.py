import streamlit as st
import requests
import tempfile
import os

st.set_page_config(page_title="Minimal RAG PDF Q&A")
st.title("Minimal RAG PDF Q&A")

backend_url = st.text_input("Backend URL", value="http://127.0.0.1:8000")
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
question = st.text_input("Question")

if st.button("Ask"):
    if not uploaded_file:
        st.error("Please upload a PDF file.")
    elif not question:
        st.error("Please enter a question.")
    else:
        with st.spinner("Uploading and querying..."):
            # save uploaded file to a temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getbuffer())
                tmp_path = tmp.name

            try:
                with open(tmp_path, "rb") as f:
                    files = {"file": (os.path.basename(tmp_path), f, "application/pdf")}
                    data = {"question": question}
                    resp = requests.post(f"{backend_url}/ask_pdf", files=files, data=data, timeout=120)

                if resp.ok:
                    try:
                        ans = resp.json().get("answer")
                    except Exception:
                        ans = resp.text
                    st.success("Answer")
                    st.write(ans)
                else:
                    st.error(f"Server error {resp.status_code}")
                    st.text(resp.text)
            except requests.exceptions.RequestException as e:
                st.error(f"Request failed: {e}")
            finally:
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

st.markdown("---")
st.markdown("Instructions: set the Backend URL if different, upload a PDF and type a question, then click Ask.")
