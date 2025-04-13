import os
import time
import datetime
import threading
import queue
import cv2
import pygame
import speech_recognition as sr
import google.generativeai as genai
from elevenlabs.client import ElevenLabs
from elevenlabs import save
from dotenv import load_dotenv
import pystray
from pystray import MenuItem as item
from PIL import Image

# --- Constants ---
STATUS_WAITING = "waiting"
STATUS_LISTENING = "listening"
STATUS_PROCESSING = "processing"
STATUS_SPEAKING = "speaking"
STATUS_ERROR = "error"

# --- Configuration Loading ---
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "pNInz6obpgDQGcFmaJgB")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gemini-2.0-flash-lite")
MIN_FACE_SIZE_PERCENT = int(os.getenv("MIN_FACE_SIZE_PERCENT", "10"))

MICROPHONE_INDEX = None
MICROPHONE_INDEX_STR = os.getenv("MICROPHONE_INDEX")
if MICROPHONE_INDEX_STR:
    try:
        MICROPHONE_INDEX = int(MICROPHONE_INDEX_STR)
        print(f"Using microphone index from .env: {MICROPHONE_INDEX}")
    except ValueError:
        print(f"Warning: Invalid MICROPHONE_INDEX '{MICROPHONE_INDEX_STR}'. Using default microphone.")

DATA_DIR = "data"
IMG_DIR = "img"
AUDIO_FILE_PATH = os.path.join(DATA_DIR, "response.mp3")
HAAR_CASCADE_PATH = os.path.join(DATA_DIR, "haarcascade_frontalface_default.xml")
INPUT_AUDIO_PATH = os.path.join(DATA_DIR, "input_audio.mp3")

ICON_PATHS = {
    STATUS_WAITING: os.path.join(IMG_DIR, "icon_waiting.png"),
    STATUS_LISTENING: os.path.join(IMG_DIR, "icon_listening.png"),
    STATUS_PROCESSING: os.path.join(IMG_DIR, "icon_processing.png"),
    STATUS_SPEAKING: os.path.join(IMG_DIR, "icon_speaking.png"),
    STATUS_ERROR: os.path.join(IMG_DIR, "icon_error.png"),
}

app_status = STATUS_WAITING
status_lock = threading.Lock()
trigger_queue = queue.Queue(maxsize=1)
stop_event = threading.Event()
tray_icon = None
loaded_icons = {}

# --- Initialization ---
try:
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY not found in .env file.")
    if not ELEVENLABS_API_KEY:
        raise ValueError("ELEVENLABS_API_KEY not found in .env file.")

    genai.configure(api_key=GOOGLE_API_KEY)
    llm_model = genai.GenerativeModel(LLM_MODEL_NAME)
    eleven_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
    recognizer = sr.Recognizer()

    print("\n--- Available Microphones ---")
    try:
        mic_names = sr.Microphone.list_microphone_names()
        if mic_names:
            for index, name in enumerate(mic_names):
                print(f"Index {index}: {name}")
        else:
            print("No microphones found.")
    except Exception as e:
        print(f"Could not list microphones: {e}")
    print("---------------------------\n")

    if not os.path.exists(HAAR_CASCADE_PATH):
        raise FileNotFoundError(f"Haar Cascade file not found at: {HAAR_CASCADE_PATH}")
    face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)

    # Create directories if missing
    for folder in (IMG_DIR, DATA_DIR):
        if not os.path.exists(folder):
            print(f"Creating directory: {folder}")
            os.makedirs(folder)

    # Load icons
    print("Loading icons...")
    for status, path in ICON_PATHS.items():
        if not os.path.exists(path):
            print(f"Warning: Icon file not found at: {path}.")
            color = 'red' if status == STATUS_ERROR else 'grey'
            loaded_icons[status] = Image.new('RGB', (64, 64), color=color)
        else:
            try:
                loaded_icons[status] = Image.open(path)
            except Exception as e:
                print(f"Error loading icon {path}: {e}")
                color = 'red' if status == STATUS_ERROR else 'grey'
                loaded_icons[status] = Image.new('RGB', (64, 64), color=color)

    if not loaded_icons.get(STATUS_WAITING):
        raise RuntimeError("Essential 'waiting' icon could not be loaded or created.")
    if not loaded_icons.get(STATUS_ERROR):
        print("Warning: 'error' icon missing. Using waiting icon as fallback.")
        loaded_icons[STATUS_ERROR] = loaded_icons.get(STATUS_WAITING)

except Exception as e:
    print(f"Initialization Error: {e}")
    exit()

# --- Status Management ---
def set_status(new_status):
    global app_status, tray_icon
    with status_lock:
        if app_status == new_status:
            return
        app_status = new_status
        print(f"Status changed to: {app_status}")
        if tray_icon and new_status in loaded_icons:
            try:
                tray_icon.icon = loaded_icons[new_status]
                tray_icon.title = f"Voice Assistant: {new_status.capitalize()}"
            except Exception as e:
                print(f"Error updating tray icon: {e}")
        elif tray_icon:
            print(f"Warning: No icon loaded for status '{new_status}'")
            tray_icon.icon = loaded_icons.get(STATUS_ERROR, tray_icon.icon)
            tray_icon.title = f"Voice Assistant: Unknown Status ({new_status})"

def get_current_status():
    with status_lock:
        return app_status

# --- Speech Processing ---
def record_and_recognize_speech():
    set_status(STATUS_LISTENING)
    text = None
    audio = None
    source_description = "microphone"

    try:
        if os.path.exists(INPUT_AUDIO_PATH):
            source_description = f"file '{INPUT_AUDIO_PATH}'"
            print(f"Loading audio from file: {INPUT_AUDIO_PATH}")
            with sr.AudioFile(INPUT_AUDIO_PATH) as source:
                audio = recognizer.record(source)
            timestamp = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
            base, ext = os.path.splitext(INPUT_AUDIO_PATH)
            processed_filename = f"{base}_processed_{timestamp}{ext}"
            os.rename(INPUT_AUDIO_PATH, processed_filename)
            print(f"Renamed input file to: {processed_filename}")
        else:
            print(f"No input file at {INPUT_AUDIO_PATH}. Using microphone.")
            try:
                mic = sr.Microphone(device_index=MICROPHONE_INDEX)
                with mic as source:
                    recognizer.adjust_for_ambient_noise(source, duration=1.0)
                    audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
            except (ValueError, OSError) as e:
                print(f"Error accessing microphone: {e}")
                set_status(STATUS_ERROR)
                return None
        if audio:
            timestamp = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
            output_recording_path = os.path.join(DATA_DIR, f"recorded_audio_{timestamp}.wav")
            with open(output_recording_path, "wb") as f:
                f.write(audio.get_wav_data())
            print(f"Recorded audio saved to {output_recording_path}")
        print("Recognizing speech...")
        set_status(STATUS_PROCESSING)
        if not audio:
            raise sr.UnknownValueError("No audio data available for recognition.")
        text = recognizer.recognize_google(audio)
        print(f"You said: {text}")
    except sr.UnknownValueError as uve:
        print(f"Speech recognition error: {uve}")
    except sr.RequestError as e:
        print(f"Request error: {e}")
        set_status(STATUS_ERROR)
    except Exception as e:
        print(f"Error in record_and_recognize_speech: {e}")
        set_status(STATUS_ERROR)
    finally:
        if text is None and get_current_status() != STATUS_ERROR:
            set_status(STATUS_WAITING)
    return text

def get_llm_response(prompt_text):
    if not prompt_text:
        return "I didn't catch that. Could you please repeat?"
    print("Sending text to Google AI...")
    set_status(STATUS_PROCESSING)
    try:
        response = llm_model.generate_content(prompt_text)
        if not response.parts:
            if response.prompt_feedback.block_reason:
                print(f"LLM request blocked: {response.prompt_feedback.block_reason}")
                return "I cannot respond to that due to safety restrictions."
            return "I received an empty response from the AI."
        return response.text
    except Exception as e:
        print(f"Error communicating with Google AI: {e}")
        set_status(STATUS_ERROR)
        return None

def generate_and_play_speech(text_to_speak):
    if not text_to_speak:
        print("No text provided for synthesis.")
        set_status(STATUS_WAITING)
        return
    print("Generating speech with ElevenLabs...")
    set_status(STATUS_PROCESSING)
    try:
        audio_data = eleven_client.generate(
            text=text_to_speak,
            voice=ELEVENLABS_VOICE_ID,
            model="eleven_multilingual_v2"
        )
        save(audio_data, AUDIO_FILE_PATH)
        print(f"Audio saved to {AUDIO_FILE_PATH}")
    except Exception as e:
        print(f"Error during TTS generation: {e}")
        set_status(STATUS_ERROR)
        if os.path.exists(AUDIO_FILE_PATH):
            try:
                os.remove(AUDIO_FILE_PATH)
            except Exception:
                pass
        return

    set_status(STATUS_SPEAKING)
    mixer_initialized = False
    try:
        pygame.mixer.init()
        mixer_initialized = True
        pygame.mixer.music.load(AUDIO_FILE_PATH)
        print("Playing audio response...")
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy() and not stop_event.is_set():
            time.sleep(0.1)
        if stop_event.is_set():
            pygame.mixer.music.stop()
        print("Playback finished.")
    except Exception as e:
        print(f"Playback error: {e}")
        set_status(STATUS_ERROR)
    finally:
        if mixer_initialized:
            try:
                pygame.mixer.music.stop()
                pygame.mixer.quit()
                print("Pygame mixer quit.")
            except Exception as pqe:
                print(f"Error quitting pygame mixer: {pqe}")
        if os.path.exists(AUDIO_FILE_PATH):
            try:
                time.sleep(0.2)
                os.remove(AUDIO_FILE_PATH)
                print(f"Cleaned up {AUDIO_FILE_PATH}")
            except Exception as cleanup_error:
                print(f"Error cleaning up audio file: {cleanup_error}")
    if get_current_status() != STATUS_ERROR and not stop_event.is_set():
        set_status(STATUS_WAITING)

# --- Face Detection ---
def face_detection_thread():
    print("Face detection thread started.")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video capture device.")
        set_status(STATUS_ERROR)
        stop_event.set()
        return
    frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640
    min_face_width = int(frame_width * (MIN_FACE_SIZE_PERCENT / 100.0))
    print(f"Minimum face width: {min_face_width} pixels")
    detection_active = True

    try:
        while not stop_event.is_set():
            if get_current_status() == STATUS_WAITING and detection_active:
                ret, frame = cap.read()
                if not ret:
                    time.sleep(1)
                    continue
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.equalizeHist(gray)
                faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(min_face_width, min_face_width))
                if any(w >= min_face_width for (_, _, w, _) in faces):
                    print("Face detected, triggering listener...")
                    try:
                        trigger_queue.put_nowait(True)
                        detection_active = False
                    except queue.Full:
                        pass
                time.sleep(0.1)
            elif get_current_status() == STATUS_WAITING and not detection_active:
                print("Resuming face detection.")
                detection_active = True
                time.sleep(0.5)
            else:
                time.sleep(0.5)
    except Exception as e:
        print(f"Error in face detection thread: {e}")
        set_status(STATUS_ERROR)
        stop_event.set()
    finally:
        cap.release()
        print("Face detection thread finished.")

# --- System Tray ---
def quit_app(icon, query):
    print("Quit command received from tray icon.")
    stop_event.set()
    try:
        while not trigger_queue.empty():
            trigger_queue.get_nowait()
    except Exception as e:
        print(f"Error clearing trigger queue: {e}")
    if icon:
        icon.stop()

def setup_tray_icon():
    global tray_icon
    menu = (item('Quit', quit_app),)
    initial_icon = loaded_icons.get(STATUS_WAITING, loaded_icons.get(STATUS_ERROR))
    if not initial_icon:
        print("FATAL: Cannot start tray icon without a valid icon.")
        stop_event.set()
        return
    tray_icon = pystray.Icon("voice_assistant", initial_icon, "Voice Assistant: Waiting", menu)
    print("Starting system tray icon...")
    try:
        tray_icon.run()
    except Exception as e:
        print(f"Error running tray icon: {e}")
        stop_event.set()

# --- Main Logic ---
def main_logic_thread():
    print("Main logic thread started.")
    while not stop_event.is_set():
        if get_current_status() == STATUS_ERROR:
            time.sleep(1)
            continue
        try:
            trigger = trigger_queue.get(timeout=1)
            if trigger and get_current_status() != STATUS_ERROR:
                user_input = record_and_recognize_speech()
                if stop_event.is_set():
                    break
                if user_input and get_current_status() != STATUS_ERROR:
                    if user_input.lower() in ["exit", "quit", "stop", "goodbye"]:
                        print("Exit command detected.")
                        generate_and_play_speech("Goodbye!")
                        time.sleep(1)
                        quit_app(tray_icon, None)
                        break
                    llm_output = get_llm_response(user_input)
                    if stop_event.is_set():
                        break
                    if llm_output and get_current_status() != STATUS_ERROR:
                        generate_and_play_speech(llm_output)
                    else:
                        print("LLM returned no output.")
                        set_status(STATUS_WAITING)
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Error in main logic loop: {e}")
            set_status(STATUS_ERROR)
            time.sleep(2)
    print("Main logic thread finished.")

# --- Main Execution ---
if __name__ == "__main__":
    print("Starting Advanced Voice Assistant...")
    face_thread = threading.Thread(target=face_detection_thread, daemon=True)
    logic_thread = threading.Thread(target=main_logic_thread, daemon=True)
    face_thread.start()
    logic_thread.start()
    setup_tray_icon()
    print("Application shutting down...")
    stop_event.set()
    print("Waiting for threads to finish...")
    face_thread.join(timeout=3)
    logic_thread.join(timeout=5)
    if face_thread.is_alive():
        print("Warning: Face detection thread did not terminate cleanly.")
    if logic_thread.is_alive():
        print("Warning: Main logic thread did not terminate cleanly.")
    if os.path.exists(AUDIO_FILE_PATH):
        try:
            os.remove(AUDIO_FILE_PATH)
            print(f"Cleaned up {AUDIO_FILE_PATH} on final exit.")
        except Exception as e:
            print(f"Could not remove {AUDIO_FILE_PATH} on final exit: {e}")
    print("Application finished.")
