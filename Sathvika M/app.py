import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import pdfplumber
from fpdf import FPDF
import time

# Load the model and tokenizer
@st.cache_resource
def load_model():
    model = AutoModelForSeq2SeqLM.from_pretrained("/content/trained_model")
    tokenizer = AutoTokenizer.from_pretrained("/content/trained_model")
    return model, tokenizer

model, tokenizer = load_model()

# Title of the app
st.title("Language Translation App")

# Tabbed Interface: Choose between text or PDF translation
tab1, tab2 = st.tabs(["Text Translation", "PDF Translation"])

# Tab 1: Translate text
with tab1:
    st.header("Translate English Text to French")
    input_text = st.text_area("Enter English text here:")
    
    # Add unique key to the slider for text translation
    accuracy = st.slider("Choose Translation Accuracy", min_value=1, max_value=10, value=4, step=1, key="text_accuracy_slider")
    
    if st.button("Translate Text"):
        if input_text.strip():
            # Perform translation with dynamic beams based on accuracy
            inputs = tokenizer.encode(input_text, return_tensors="pt")
            outputs = model.generate(inputs, max_length=256, num_beams=accuracy, early_stopping=True)
            translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
            st.success(translation)

            # Provide the option to download the translated text as a PDF
            translated_pdf = FPDF()
            translated_pdf.add_page()
            translated_pdf.set_font("Arial", size=12)
            translated_pdf.multi_cell(0, 10, translation)
            pdf_path = "/content/translated_text_output.pdf"
            translated_pdf.output(pdf_path)

            with open(pdf_path, "rb") as f:
                st.download_button(
                    label="Download Translated Text as PDF",
                    data=f,
                    file_name="translated_text_output.pdf",
                    mime="application/pdf"
                )
        else:
            st.error("Please enter some text.")

# Tab 2: Translate PDF
with tab2:
    st.header("Upload a PDF in English")
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")

    if uploaded_file is not None:
        # Read the PDF content using pdfplumber
        text = ""
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                text += page.extract_text()

        # Ensure no broken words or spacing issues in the extracted text
        text = text.replace("\n", " ").strip()  # Remove newlines and extra spaces
        
        st.write("Original Text:")
        st.text_area("Extracted Text from PDF", text, height=200)

        # Add unique key to the slider for PDF translation
        accuracy = st.slider("Choose Translation Accuracy", min_value=1, max_value=10, value=4, step=1, key="pdf_accuracy_slider")

        # Option for translation time execution
        if st.button("Translate PDF"):
            if text.strip():
                # Start time tracking for execution
                start_time = time.time()
                
                # Perform translation with dynamic beams based on accuracy
                inputs = tokenizer.encode(text, return_tensors="pt", truncation=True)
                outputs = model.generate(inputs, max_length=1024, num_beams=accuracy, early_stopping=True)
                translation = tokenizer.decode(outputs[0], skip_special_tokens=True)

                # End time tracking
                end_time = time.time()
                execution_time = end_time - start_time  # Time taken for translation

                st.write(f"Translation completed in {execution_time:.2f} seconds.")
                st.success(translation)

                # Provide the option to download the translated text as a PDF
                translated_pdf = FPDF()
                translated_pdf.add_page()
                translated_pdf.set_font("Arial", size=12)
                translated_pdf.multi_cell(0, 10, translation)
                pdf_path = "/content/translated_pdf_output.pdf"
                translated_pdf.output(pdf_path)

                with open(pdf_path, "rb") as f:
                    st.download_button(
                        label="Download Translated PDF",
                        data=f,
                        file_name="translated_pdf_output.pdf",
                        mime="application/pdf"
                    )
            else:
                st.error("The uploaded PDF is empty!")
