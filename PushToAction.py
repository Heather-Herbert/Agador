
import speech_recognition as sr
import os

def action(command):
    os.system(command)

r = sr.Recognizer()
mic = sr.Microphone()

with mic as source:
    r.adjust_for_ambient_noise(source)
    audio = r.listen(source)
    action(r.recognize_google(audio))

