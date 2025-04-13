import speech_recognition as sr
import google.generativeai as genai
from elevenlabs.client import ElevenLabs
from elevenlabs import save
import pygame
import os
import time
import threading
import queue
import cv2 # OpenCV
from dotenv import load_dotenv # For .env file
import pystray # Use pystray
from pystray import MenuItem as item # For pystray menu
from PIL import Image

# --- Constants ---
STATUS_WAITING = "waiting"
STATUS_LISTENING = "listening"
STATUS_PROCESSING = "processing"
STATUS_SPEAKING = "speaking"
STATUS_ERROR = "error"

# --- Configuration Loading ---
load_dotenv() # Load variables from .env file

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "pNInz6obpgDQGcFmaJgB") # Default if not set
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gemini-pro")
MIN_FACE_SIZE_PERCENT = int(os.getenv("MIN_FACE_SIZE_PERCENT", "10")) # Default 10%

# --- Global Variables & Shared Resources ---
AUDIO_FILE_PATH = "data/response.mp3"
HAAR_CASCADE_PATH = "data/haarcascade_frontalface_default.xml" # Expect file in same dir
ICON_PATHS = {
    STATUS_WAITING: "img/icon_waiting.png",
    STATUS_LISTENING: "img/icon_listening.png",
    STATUS_PROCESSING: "img/icon_processing.png",
    STATUS_SPEAKING: "img/icon_speaking.png",
    STATUS_ERROR: "img/icon_error.png",
}

app_status = STATUS_WAITING
status_lock = threading.Lock() # To safely update status from different threads
trigger_queue = queue.Queue(maxsize=1) # To signal listening trigger from CV thread
stop_event = threading.Event() # To signal threads to stop
tray_icon = None # Global reference to the pystray icon instance
loaded_icons = {} # Dictionary to hold loaded PIL Image objects

# --- Initialization ---
try:
    # Check for API Keys early
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY not found in .env file.")
    if not ELEVENLABS_API_KEY:
        raise ValueError("ELEVENLABS_API_KEY not found in .env file.")

    # Configure Google AI
    genai.configure(api_key=GOOGLE_API_KEY)
    llm_model = genai.GenerativeModel(LLM_MODEL_NAME)

    # Configure ElevenLabs
    eleven_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

    # Initialize Speech Recognizer
    recognizer = sr.Recognizer()

    # Load OpenCV Face Detector
    if not os.path.exists(HAAR_CASCADE_PATH):
        raise FileNotFoundError(f"Haar Cascade file not found at: {HAAR_CASCADE_PATH}")
    face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)

    # --- Load Icons using Pillow for pystray ---
    print("Loading icons...")
    for status, path in ICON_PATHS.items():
        if not os.path.exists(path):
            print(f"Warning: Icon file not found at: {path}. Skipping.")
            # Create a placeholder if needed, especially for error/waiting
            if status in [STATUS_WAITING, STATUS_ERROR]:
                 loaded_icons[status] = Image.new('RGB', (64, 64), color = 'red' if status == STATUS_ERROR else 'grey')
                 print(f"Created placeholder icon for {status}")
        else:
            try:
                loaded_icons[status] = Image.open(path)
            except Exception as e:
                 print(f"Error loading icon {path}: {e}")
                 if status in [STATUS_WAITING, STATUS_ERROR]:
                      loaded_icons[status] = Image.new('RGB', (64, 64), color = 'red' if status == STATUS_ERROR else 'grey')
                      print(f"Created placeholder icon for {status} due to load error.")

    # Ensure we have essential icons
    if not loaded_icons.get(STATUS_WAITING):
         raise RuntimeError("Essential 'waiting' icon could not be loaded or created.")
    if not loaded_icons.get(STATUS_ERROR):
         print("Warning: Error icon missing or failed to load. Using waiting icon as fallback.")
         loaded_icons[STATUS_ERROR] = loaded_icons.get(STATUS_WAITING)


except ValueError as ve:
    print(f"Configuration Error: {ve}")
    exit()
except FileNotFoundError as fnfe:
    print(f"Initialization Error: {fnfe}")
    exit()
except RuntimeError as rte:
     print(f"Initialization Error: {rte}")
     exit()
except Exception as e:
    print(f"Unexpected Error during initialization: {e}")
    exit()

# Removed wxPython TaskBarIcon Class

# --- Status Management ---
def set_status(new_status):
    """Safely updates the global application status and tray icon."""
    global app_status, tray_icon # Need tray_icon instance
    with status_lock:
        if app_status == new_status: # Avoid unnecessary updates
             return
        app_status = new_status
        print(f"Status changed to: {app_status}")
        # --- Update pystray icon (pystray handles thread safety for icon property) ---
        if tray_icon and new_status in loaded_icons:
            try:
                tray_icon.icon = loaded_icons[new_status]
                tray_icon.title = f"Voice Assistant: {new_status.capitalize()}" # Update tooltip
            except Exception as e:
                # This might happen if the app is shutting down rapidly
                print(f"Error updating tray icon property: {e}")
        elif tray_icon:
             print(f"Warning: No icon loaded for status '{new_status}'")
             # Optionally set a default/error icon
             if STATUS_ERROR in loaded_icons:
                 tray_icon.icon = loaded_icons[STATUS_ERROR]
                 tray_icon.title = f"Voice Assistant: Unknown Status ({new_status})"


# --- Core Functions (Largely unchanged, ensure status updates work) ---

def record_and_recognize_speech():
    """Records audio and converts to text. Sets status."""
    set_status(STATUS_LISTENING)
    text = None
    mic = None # Initialize mic variable
    try:
        mic = sr.Microphone()
        with mic as source:
            print("Adjusting for ambient noise...")
            recognizer.adjust_for_ambient_noise(source, duration=0.5) # Shorter adjustment
            print("Listening for command...")
            try:
                # Use listen with a timeout, check stop_event periodically if needed
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
            except sr.WaitTimeoutError:
                print("No speech detected within the timeout period.")
                set_status(STATUS_WAITING) # Go back to waiting if timeout
                return None # Return None if timeout

        print("Recognizing speech...")
        set_status(STATUS_PROCESSING) # Recognize = Processing
        try:
            text = recognizer.recognize_google(audio)
            print(f"You said: {text}")
        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand audio.")
            text = None # Ensure text is None if not recognized
        except sr.RequestError as e:
            print(f"Could not request results from Google Service; {e}")
            set_status(STATUS_ERROR) # Indicate error
            text = None
        except Exception as e:
            print(f"An unexpected error occurred during speech recognition: {e}")
            set_status(STATUS_ERROR)
            text = None

    except Exception as e:
        print(f"Error accessing microphone: {e}")
        set_status(STATUS_ERROR)
        text = None
    finally:
        # If text is None after trying, reset status (unless already error)
        if text is None and get_current_status() != STATUS_ERROR:
             set_status(STATUS_WAITING)
    return text


def get_llm_response(prompt_text):
    """Gets response from LLM. Sets status."""
    if not prompt_text:
        return "I didn't catch that. Could you please repeat?"

    print("Sending text to Google AI...")
    set_status(STATUS_PROCESSING)
    llm_text = "Sorry, I encountered an error processing your request." # Default error message
    try:
        response = llm_model.generate_content(prompt_text)
        if not response.parts:
            if response.prompt_feedback.block_reason:
                print(f"LLM request blocked: {response.prompt_feedback.block_reason}")
                llm_text = "I cannot respond to that due to safety restrictions."
            else:
                 llm_text = "I received an empty response from the AI."
        else:
            llm_text = response.text
        print(f"AI responded: {llm_text}")
    except Exception as e:
        print(f"Error communicating with Google AI: {e}")
        set_status(STATUS_ERROR)
        llm_text = None # Indicate failure
    finally:
        # Status will be set to SPEAKING or WAITING after this by the caller
        pass
    return llm_text

def generate_and_play_speech(text_to_speak):
    """Generates and plays speech. Sets status."""
    if not text_to_speak:
        print("No text provided to synthesize.")
        set_status(STATUS_WAITING) # Go back to waiting if nothing to say
        return

    print("Generating speech with ElevenLabs...")
    set_status(STATUS_PROCESSING) # Generation is processing
    audio_generated = False
    try:
        # --- Generate Audio ---
        audio_data = eleven_client.generate(
            text=text_to_speak,
            voice=ELEVENLABS_VOICE_ID,
            model="eleven_multilingual_v2"
        )
        # --- Save Audio ---
        save(audio_data, AUDIO_FILE_PATH)
        print(f"Audio saved to {AUDIO_FILE_PATH}")
        audio_generated = True

    except Exception as e:
        print(f"Error during ElevenLabs TTS generation: {e}")
        set_status(STATUS_ERROR)
        # Clean up potentially incomplete file
        if os.path.exists(AUDIO_FILE_PATH):
            try: os.remove(AUDIO_FILE_PATH)
            except Exception: pass
        return # Don't proceed to playback

    # --- Playback ---
    if audio_generated:
        set_status(STATUS_SPEAKING) # Set status before playing
        mixer_initialized = False
        try:
            pygame.mixer.init() # Initialize mixer just before playback
            mixer_initialized = True
            pygame.mixer.music.load(AUDIO_FILE_PATH)
            print("Playing audio response...")
            pygame.mixer.music.play()

            # Wait for playback completion or stop signal
            while pygame.mixer.music.get_busy() and not stop_event.is_set():
                time.sleep(0.1)

            if stop_event.is_set():
                print("Playback interrupted by stop signal.")
                pygame.mixer.music.stop()
            else:
                print("Playback finished.")

        except Exception as e:
            print(f"Error during audio playback: {e}")
            set_status(STATUS_ERROR)
        finally:
            if mixer_initialized:
                try:
                    pygame.mixer.music.stop() # Ensure music is stopped
                    pygame.mixer.quit() # Quit mixer after playback
                    print("Pygame mixer quit.")
                except Exception as pqe:
                    print(f"Error quitting pygame mixer: {pqe}")

            # --- Clean up Audio File ---
            if os.path.exists(AUDIO_FILE_PATH):
                try:
                    time.sleep(0.2) # Small delay before removing
                    os.remove(AUDIO_FILE_PATH)
                    print(f"Cleaned up {AUDIO_FILE_PATH}")
                except Exception as cleanup_error:
                    # This can sometimes fail if the file lock isn't released immediately
                    print(f"Warning: Error cleaning up audio file: {cleanup_error}")

    # --- Reset Status ---
    # Set status back to waiting only if no error occurred during playback/cleanup
    # and if the app isn't stopping
    current_status = get_current_status()
    if current_status != STATUS_ERROR and not stop_event.is_set():
         set_status(STATUS_WAITING)


# --- Face Detection Thread ---
def face_detection_thread():
    """Thread to monitor camera for a face looking."""
    print("Face detection thread started.")
    cap = None
    min_face_width = 0
    detection_active = True # Control detection loop within the thread

    try:
        cap = cv2.VideoCapture(0) # 0 is usually the default webcam
        if not cap.isOpened():
            print("Error: Could not open video capture device.")
            set_status(STATUS_ERROR)
            stop_event.set() # Signal other threads to stop too
            return

        frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        if frame_width == 0: frame_width = 640 # Default
        min_face_width = int(frame_width * (MIN_FACE_SIZE_PERCENT / 100.0))
        print(f"Minimum face width to trigger: {min_face_width} pixels")

        while not stop_event.is_set():
            current_status = get_current_status()

            if current_status == STATUS_WAITING and detection_active:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Failed to capture frame.")
                    time.sleep(1); continue

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.equalizeHist(gray)
                faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(min_face_width, min_face_width))

                found_face = any(w >= min_face_width for (x, y, w, h) in faces)

                if found_face:
                    print("Face detected, triggering listener...")
                    try:
                        trigger_queue.put_nowait(True)
                        detection_active = False # Pause detection
                    except queue.Full: pass # Ignore if busy
                    except Exception as e: print(f"Error putting item in queue: {e}")

                time.sleep(0.1) # Short sleep

            elif not detection_active and current_status == STATUS_WAITING:
                 print("Resuming face detection.")
                 detection_active = True
                 time.sleep(0.5) # Small delay before resuming
            else:
                time.sleep(0.5) # Sleep longer if not waiting/detecting

    except Exception as e:
        print(f"Error in face detection thread: {e}")
        set_status(STATUS_ERROR)
        stop_event.set() # Signal exit on error
    finally:
        if cap: cap.release()
        print("Face detection thread finished.")


# --- System Tray Functions ---
def get_current_status():
    """Safely get the current status."""
    with status_lock:
        return app_status

def quit_app(icon, query):
    """Callback function to stop the application from pystray menu."""
    print("Quit command received from tray icon.")
    stop_event.set() # Signal all threads to stop
    # Attempt to clear the trigger queue
    try:
        while not trigger_queue.empty():
            trigger_queue.get_nowait()
    except queue.Empty: pass
    except Exception as e: print(f"Error clearing trigger queue on exit: {e}")

    if icon:
        icon.stop() # Stop the pystray icon loop

def setup_tray_icon():
    """Creates and runs the pystray system tray icon."""
    global tray_icon # Allow modification of the global instance
    menu = (item('Quit', quit_app),)
    initial_icon = loaded_icons.get(STATUS_WAITING, loaded_icons.get(STATUS_ERROR)) # Use waiting or error icon
    if not initial_icon:
         print("FATAL: Cannot start tray icon without a valid initial icon.")
         return # Cannot proceed

    tray_icon = pystray.Icon(
        "voice_assistant",
        initial_icon,
        "Voice Assistant: Waiting", # Initial tooltip
        menu)
    print("Starting system tray icon...")
    # This blocks the main thread until icon.stop() is called
    try:
        tray_icon.run()
    except Exception as e:
        print(f"Error running tray icon: {e}")
        stop_event.set() # Ensure other threads stop if tray crashes


# --- Main Application Logic Thread ---
def main_logic_thread():
    """Handles the main interaction flow when triggered."""
    print("Main logic thread started.")
    while not stop_event.is_set():
        try:
            trigger = trigger_queue.get(timeout=1) # Wait for trigger

            if trigger and get_current_status() != STATUS_ERROR:
                user_input = record_and_recognize_speech()
                if stop_event.is_set(): break

                current_status = get_current_status() # Re-check status
                if user_input and current_status != STATUS_ERROR:
                    if user_input.lower() in ["exit", "quit", "stop", "goodbye"]:
                        print("Exit command detected.")
                        generate_and_play_speech("Goodbye!")
                        time.sleep(1)
                        # Signal exit via the quit_app function (doesn't need CallAfter now)
                        quit_app(tray_icon, None) # Trigger shutdown
                        break

                    llm_output = get_llm_response(user_input)
                    if stop_event.is_set(): break

                    current_status = get_current_status() # Re-check status
                    if llm_output and current_status != STATUS_ERROR:
                        generate_and_play_speech(llm_output)
                    elif current_status != STATUS_ERROR:
                        set_status(STATUS_WAITING) # Reset if LLM failed okay
                # If speech recognition failed, status is reset inside its function

            trigger_queue.task_done() # Mark trigger processed

        except queue.Empty:
            continue # Expected timeout
        except Exception as e:
            print(f"Error in main logic loop: {e}")
            set_status(STATUS_ERROR)
            time.sleep(2)

    print("Main logic thread finished.")


# --- Main Execution ---
if __name__ == "__main__":
    # Removed wx.App initialization

    print("Starting Advanced Voice Assistant (pystray)...")

    # --- Start Background Threads ---
    face_thread = threading.Thread(target=face_detection_thread, daemon=True)
    face_thread.start()

    logic_thread = threading.Thread(target=main_logic_thread, daemon=True)
    logic_thread.start()

    # --- Start pystray Event Loop ---
    # This function now blocks the main thread until quit_app calls icon.stop()
    setup_tray_icon()

    # --- Cleanup (after pystray loop finishes) ---
    print("Application shutting down...")
    # stop_event should already be set by the Quit handler
    if not stop_event.is_set():
        stop_event.set() # Ensure it's set if loop exited unexpectedly

    print("Waiting for threads to finish...")
    face_thread.join(timeout=2)
    logic_thread.join(timeout=5)

    # Final audio file cleanup
    if os.path.exists(AUDIO_FILE_PATH):
        try:
            os.remove(AUDIO_FILE_PATH)
            print(f"Cleaned up {AUDIO_FILE_PATH} on final exit.")
        except Exception as e:
            print(f"Could not remove {AUDIO_FILE_PATH} on final exit: {e}")

    print("Application finished.")

