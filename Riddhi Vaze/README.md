# **Neural Machine Translator (NMT)**

## **Overview**
The **Neural Machine Translator (NMT)** project is a user-friendly application designed to facilitate the translation of English text into multiple target languages, including **French**, **Gujarati**, **Hindi**, and **Marathi**. Built using state-of-the-art neural machine translation models and an interactive user interface powered by **Gradio**, this project aims to make language translation accessible and efficient for users of diverse linguistic backgrounds.

---

## **Key Features**
1. **Multiple Input Methods:**
   - **Manual Text Input:** Users can type English text directly into the application.
   - **File Upload:** Users can upload `.txt`, `.docx`, or `.pdf` files containing English text for translation.

2. **Language Options:**
   - The application supports translation into **French**, **Gujarati**, **Hindi**, and **Marathi**, catering to both global and regional language needs.

3. **Translation Accuracy:**
   - Powered by the **Facebook M2M100 Neural Machine Translation model**, ensuring high-quality translations for supported languages.

4. **Dynamic Interface:**
   - The interface adapts based on user preferences (manual text input or file upload), providing a seamless and intuitive experience.

5. **Interactive Design:**
   - The interface features a visually appealing layout with a light-yellow theme and bold headings for enhanced readability and usability.

---

## **Technical Stack**
- **Gradio**: Provides the interactive web interface for users to input text, upload files, and view translations.
- **Transformers Library**: Implements the neural machine translation models for high-quality language conversion.
- **Document Handling Libraries**:
  - **docx**: Reads `.docx` files.
  - **PyPDF2**: Extracts text from `.pdf` files.
- **Python**: The core programming language used to integrate components and create a cohesive application.

---

## **How It Works**
1. **Input Selection**: The user selects their preferred input method (manual text entry or file upload).
2. **Text Processing**:
   - If manual input is selected, the user types the text in English.
   - If file upload is chosen, the application reads and extracts the text from the uploaded file.
3. **Translation**: The application translates the English text into the selected target language using the appropriate neural model.
4. **Output**: The translated text is displayed in a dedicated output box for the user to view or copy.

---

## **Applications**
- **Education**: Assists students and educators in understanding and translating text across languages.
- **Localization**: Helps businesses and content creators translate material for multilingual audiences.
- **Accessibility**: Makes information accessible to speakers of regional languages like Gujarati, Hindi, and Marathi.

---

## **Getting Started**
1. Clone the repository to your local machine.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Run the application script to launch the Gradio interface.
4. Access the interface in your web browser and start translating!

---

## **Future Enhancements**
- Support for additional languages.
- Integration with text-to-speech for audio translations.
- Enhanced file processing for other formats like `.odt` and `.rtf`.

---

This project showcases the power of neural networks in breaking language barriers, fostering inclusivity, and promoting global communication.
