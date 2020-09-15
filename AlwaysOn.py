
import speech_recognition as sr
import os

def action(command):
    try:
        (firstchunk, rest) = command.split(maxsplit=2)
        if firstchunk == "hello computer":
            print("*")
        else:
            print(command)
            print(rest)
    except Exception as inst:
        print(type(inst))  # the exception instance
        print(inst.args)  # arguments stored in .args

r = sr.Recognizer()
mic = sr.Microphone()

while True:
    with mic as source:
        try:
            r.adjust_for_ambient_noise(source)
            audio = r.listen(source)
            action(r.recognize_google(audio))
        except Exception as inst:
            print(type(inst))    # the exception instance
            print(inst.args)     # arguments stored in .args


