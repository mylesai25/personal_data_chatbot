# MylesAI Personal Data Chatbot

This Streamlit application utilizes multiple AI models from OpenAI, Anthropic, and Anyscale to create a personalized data chatbot. It allows users to upload documents, select an AI model, and interact with the AI through a chat interface. The AI provides detailed and context-aware responses based on the uploaded document.

## Features

- **Document Upload**: Upload a PDF document for the chatbot to analyze.
- **AI Model Selection**: Choose between OpenAI, Anthropic, and Anyscale models.
- **Chat Interface**: Interactive chat interface for user queries and responses.
- **Source Display**: View the sources of the AI's responses.

## Requirements

- Python 3.8 or higher
- Streamlit
- OpenAI API key, Anthropic API key, or Anyscale API key
- Additional libraries: `llama_index`, `nltk`, `tiktoken`, `pandas`, `numpy`, `pypdf`

## Setup

1. **Clone the repository**:
    ```sh
    git clone <repository_url>
    cd <repository_directory>
    ```

2. **Install dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

3. **Get your API keys**:
    - Enter your API keys into the app:
    ```sh
    OPENAI_API_KEY
    ANTHROPIC_API_KEY
    ANYSCALE_API_KEY
    ```

4. **Run the Streamlit app**:
    ```sh
    streamlit run app.py
    ```

## Usage

1. **Select AI Provider**:
    - Use the sidebar to select the AI provider (OpenAI, Anthropic, or Anyscale).

2. **Enter API Key**:
    - Enter the respective API key in the sidebar.

3. **Upload Document**:
    - Use the sidebar to upload your PDF document.

4. **Interact with the Chatbot**:
    - Enter your chat prompts in the input box and receive responses.
    - View the sources of the AI's responses by expanding the "Sources" section.

5. **Clear Chat**:
    - Use the "Clear Chat" button in the sidebar to reset the chat history.

## Functions

### `extract_text_from_pdf(pdf_path)`
Extracts all text from a PDF file and returns it as a string.

### `get_llm(model_name, api_key=None)`
Returns the appropriate language model based on the model name and API key.

### `get_chat_engine(file, model_name)`
Initializes and returns the chat engine based on the uploaded file and selected model.

## Components

- **`llama_index`**: Core library for indexing and querying documents.
- **`nltk`**: Natural language processing library for text tokenization.
- **`tiktoken`**: Tokenization library for text processing.
- **`pypdf`**: Library for reading PDF files.
- **`streamlit`**: Web application framework for building the chatbot interface.

## Notes

- Ensure you have valid API keys for the selected AI models.
- The app caches resources like language models and chat engines for efficiency.
- The chat interface displays both user inputs and AI-generated responses, including the sources of the information.

## Contributing

Feel free to fork the repository and submit pull requests for improvements or additional features.

## License

This project is licensed under the MIT License.

---

Enjoy your personalized AI data chatbot experience!
