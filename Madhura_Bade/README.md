# English-Spanish Translation System

## Project Overview
This project is a machine translation system capable of translating text and PDF documents between English to Spanish. Users can upload a PDF or input text directly, and the system will provide translations in the desired language.

The system leverages the **MarianMT model**, a state-of-the-art machine translation framework, for accurate and efficient translations.

---

## Features
- **Text Translation**: Translate sentences or paragraphs between English and Spanish.
- **PDF Upload and Translation**: Upload PDF files to extract and translate their contents.
- **Bidirectional Translation**: Supports translations from English to Spanish and vice versa.
- **User-Friendly Interface**: Simple workflow for uploading files and retrieving translations.

---

## Technologies Used
- **Model**: MarianMT (Hugging Face Transformers)
- **Programming Language**: Python
- **Libraries**:
  - `transformers` (for MarianMT model)
  - `PyPDF2` (for PDF text extraction)

---

## How It Works
1. **Text Input**:
   - Users can enter english text into a text box for translation.
   - The system translates it to the Spanish language.

2. **PDF Upload**:
   - Users upload a PDF file.
   - Text is extracted from the PDF and passed to the MarianMT model for translation.

3. **Output**:
   - The translated text is displayed on the screen or saved as a downloadable file.

---


## Future Enhancements
- Add support for additional languages.
- Improve the user interface for seamless interaction.
- Enhance PDF parsing for complex documents.

---

## Contribution
Contributions are welcome! Feel free to fork the repository and submit pull requests.

---

## License
This project is licensed under the [MIT License](LICENSE).

---

## Acknowledgments
- Hugging Face for the MarianMT model.
- NLTK and PyPDF2 for their robust libraries.

