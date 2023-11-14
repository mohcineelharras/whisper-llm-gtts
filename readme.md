# VoiceAI whisper-llm-gtts

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## Overview

VoiceAI is an advanced AI-powered application that combines Text-to-Speech (TTS), Speech-to-Text (STT), and a Local Language Model (LLM). It empowers users to convert text to speech, transcribe audio to text, and interact with a powerful local language model, all within a seamless and intuitive interface.

## Table of Contents

- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
  - [Running the App](#running-the-app)
- [Dockerization](#dockerization)
- [Contributing](#contributing)
- [License](#license)

## Getting Started

To get started with VoiceAI, follow these simple steps:

### Prerequisites

Before you begin, ensure you have the following prerequisites:

- Python 3.10 or higher
- ideally a GPU to run LLM + Whisper confortably
- Docker
or
- env (conda or venv) with appropriate libraries

### Installation

1. Clone the VoiceAI repository:

   ```bash
   git clone https://github.com/yourusername/VoiceAI.git
   cd VoiceAI
   ```
2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

VoiceAI offers a wide range of functionalities related to text-to-speech, speech-to-text, and local language modeling. Here's how you can make the most of it:

### Running the App

To run VoiceAI locally, use the following command:

```bash
streamlit run your_app_script.py
```

Then, open a web browser and go to [http://localhost:8501](http://localhost:8501) to access the app.

## Dockerization

You can also run VoiceAI in a Docker container for added convenience. Follow these steps to get started:

### Dockerizing the App

1. Build the Docker image for VoiceAI:

   ```bash
   docker build -t voice-ai .
   ```

2. Launch the Docker container:

   ```bash
   docker run -p 80:80 voice-ai
   ```

To access the app, open a web browser and navigate to [http://localhost](http://localhost).

## Contributing

We welcome contributions from the community. If you'd like to contribute to VoiceAI, please review our [Contributing Guidelines](CONTRIBUTING.md).

## License

VoiceAI is licensed under the MIT License. For details, see the [LICENSE](LICENSE) file.
```

