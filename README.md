# Agador

Agador is a cutting-edge voice assistant application that combines speech recognition, AI-powered responses, text-to-speech synthesis, and face detection. The project leverages multiple APIs and libraries to create an interactive, voice-enabled experience.

## Features

- **Speech Recognition:** Uses the Google Speech Recognition API to convert spoken language to text.
- **Generative AI:** Utilizes the Google Generative AI API for producing natural language responses.
- **Text-to-Speech:** Synthesizes audio responses using the ElevenLabs API.
- **Face Detection:** Monitors video streams through a webcam using OpenCV to detect faces.
- **System Tray Integration:** Provides a system tray icon for quick control and graceful shutdown.
- **Audio Bypass Mode:** Supports pre-recorded audio file input, bypassing the live microphone for testing and simulation.

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/Heather-Herbert/agador.git
   cd agador
   ```

2. **Install Dependencies:**

   Ensure you have [Python 3](https://www.python.org/downloads/) installed. Install the project dependencies using `pip` and the provided `requirements.txt`:

   ```bash
   pip install -r requirements.txt
   ```

## Configuration

### .env File

Create a `.env` file in the project root directory to store your API keys and configuration settings. Below is a sample template:

```env
# Google API Key for Generative AI (Get this from the Google Cloud Console)
GOOGLE_API_KEY=your_google_api_key_here

# ElevenLabs API Key (Obtain from your ElevenLabs account)
ELEVENLABS_API_KEY=your_elevenlabs_api_key_here

# Optional: ElevenLabs Voice ID (defaults to "pNInz6obpgDQGcFmaJgB")
ELEVENLABS_VOICE_ID=your_preferred_voice_id

# Optional: LLM model name (defaults to "gemini-pro")
LLM_MODEL_NAME=your_llm_model_name

# Optional: Minimum face size percent to trigger face detection (defaults to 10)
MIN_FACE_SIZE_PERCENT=10

# Optional: Microphone index to use (leave unset to use the default microphone)
MICROPHONE_INDEX=0
```

**Obtaining API Keys:**

- **Google Generative AI:**
  - Visit the [Google Cloud Console](https://console.cloud.google.com/).
  - Create a new project or select an existing one.
  - Enable the relevant Generative AI API.
  - Generate and copy your API key.

- **ElevenLabs:**
  - Sign up or log in at [ElevenLabs](https://beta.elevenlabs.io/).
  - Navigate to the API keys section in your account settings.
  - Copy your API key and add it to your `.env` file.

## Audio Bypass Mode

Agador includes an **audio bypass mode** that allows the use of a pre-recorded audio file in place of the live microphone input. This mode is particularly useful for testing or simulating voice inputs.

### How to Use Audio Bypass Mode

1. **Prepare an Audio File:**
   - Place your audio file (supported formats such as MP3 or WAV) in the `data` directory.
   - Rename the audio file to `input_audio.mp3`.

2. **Run the Application:**
   - When Agador starts, it checks the `data` folder for the `input_audio.mp3` file.
   - If the file is found, Agador will process it in place of live audio captured from the microphone.
   - After processing, the file is renamed to prevent reprocessing in subsequent runs.

## Running the Application

After installing dependencies and setting up your `.env` file, run the application with:

```bash
python agador.py
```

Agador will then launch with a system tray icon. The application will list available microphones (if using live mode) and continuously monitor for a face through your webcam to trigger the voice recognition flow.

## Troubleshooting

- **API Keys Missing or Invalid:**  
  Ensure your `.env` file is correctly configured with valid API keys.

- **Microphone Issues:**  
  Verify that the `MICROPHONE_INDEX` is correct or leave it unset to display the option on your system.

- **Audio File Handling:**  
  Confirm that your pre-recorded audio file is named `input_audio.mp3` and located in the `data` directory if using audio bypass mode.

- **Dependencies:**  
  Double-check that all required packages from `requirements.txt` are installed.

## Contributing

Contributions are welcome! Whether you want to add new features, fix bugs, or improve documentation, please feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License.
