# BestAI - Intelligent Model Router

BestAI is a smart chat application that automatically routes your prompts to the best specialized AI model for the job. Whether you're coding, writing creatively, or generating art concepts, BestAI ensures your request is handled by the most capable model available.

## 🚀 Features

- **Smart Intent Routing**: Automatically classifies your request into categories (Coding, Writing, Art, or General) and routes it to the specialized model.
- **Multi-Model Intelligence**:
  - **Coding**: Powered by **Moonshot (Kimi-k2)** for precise logic and syntax.
  - **Writing**: Powered by **GPT-OSS 120B** for creative and nuanced text generation.
  - **Art/Visuals**: Powered by **NVIDIA Nemotron** for visual descriptions and artistic concepts.
  - **General**: Efficiently handled by optimized general-purpose models.
- **Chat History**: Automatically saves and persists your conversation history locally.
- **Advanced Tools**: Includes built-in summarization and text simplification capabilities.

## 🛠️ Technology Stack

- **Backend**: Python / Flask
- **AI Integration**: OpenAI SDK (Universal Client for NVIDIA, Moonshot, etc.)
- **Frontend**: HTML/CSS/JS (Jinja2 Templates)

## 📦 Installation

1.  **Clone the repository**
    ```bash
    git clone <repository-url>
    cd Best-ai
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    pip install openai  # Required for model clients
    ```

3.  **Environment Setup**
    Open `app.py` and ensure your API keys are configured:
    - `NVIDIA_API_KEY`
    - `MOONSHOT_API_KEY`
    - `GPTOSS120B_API_KEY`
    - `ANTHROPIC_API_KEY` (Optional)

## 🏃‍♂️ Usage

1.  **Start the Application**
    ```bash
    python app.py
    ```

2.  **Access the UI**
    Open your browser and navigate to:
    `http://127.0.0.1:5000`

3.  **Chat**
    Type your message. The **Master AI Router** will analyze your intent and seamlessly dispatch it to the best model.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
