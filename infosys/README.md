# Translation App

This repository contains a Streamlit-based Translation App built using NLP models for multilingual translation. The app supports translations among English, Spanish, French, and Italian, leveraging state-of-the-art models trained with Seq2Seq and Hensiki Optmus MT transformers.

## Features

### Language Options
- Supports multiple language pairs: `en-es`, `en-fr`, `en-it`, `it-en`, `fr-en`, and `es-en`.

### Input Flexibility
- **Text Input**: Instant translation for short texts.
- **File Upload**: Bulk translation with the option to download the translated file.

### Streamlit Interface
- User-friendly interface for selecting input and target languages.
- Provides a seamless translation experience.

## Dataset

The dataset consists of files named like `en-es_train.csv`, `en-fr_train.csv`, etc. Each file contains two columns:
- **Input Text**: The original text to be translated.
- **Translated Text**: The corresponding translation based on the file name.

## Models

Six translation models were trained using Seq2Seq and Hensiki Optmus MT transformers. Each model is tailored for a specific language pair. After training, the models were saved and imported into the Streamlit app.

## Run the App

To run the app locally, execute the following command:

```bash
streamlit run app.py
```

## Usage

1. Open the app in your browser (Streamlit will provide a local URL).
2. Select the input language and target language from the dropdown menus.
3. Provide the text input or upload a file to translate.
4. If a file is uploaded, download the translated file after processing.

### Examples

#### Text Input
- **Input Language**: English  
- **Target Language**: Spanish  
- **Input Text**: "Hello, how are you?"  
- **Output Text**: "Hola, ¿cómo estás?"

#### File Upload
- Upload a file (e.g., `sample.csv` with a column of English text).
- Select the target language.
- Download the translated file.

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests to improve the app or add new features.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
