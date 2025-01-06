import speech_recognition as sr
import pyttsx3
from datetime import datetime
from transformers import pipeline

# Initialize TTS engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Adjust speaking rate
engine.setProperty('volume', 0.9)  # Set volume level

# Initialize NLP pipeline (DistilBERT for sentiment analysis and GPT-2 for conversations)
nlp = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english", return_all_scores=False)
conversation_model = pipeline("text-generation", model="gpt2")

# Global variables for medication tracking
medication_reminder_time = datetime.now().replace(hour=14, minute=0, second=0, microsecond=0)  # Set reminder for 2 PM
medication_taken = False  # Flag to track if the medication has been taken

# Text-to-Speech Function
def speak(text):
    print(f"System: {text}")  # Debugging/logging
    engine.say(text)
    engine.runAndWait()

# Speech-to-Text Function
def listen():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source, duration=1)  # Adjust to background noise
        try:
            audio = recognizer.listen(source, timeout=5)
            text = recognizer.recognize_google(audio).lower()
            return text
        except sr.UnknownValueError:
            speak("Sorry, I didn't catch that. Can you repeat?")
            return ""
        except sr.RequestError:
            speak("There was an error with the speech recognition service.")
            return ""

# Function to Handle Medication Reminders
def handle_medication_reminder():
    global medication_taken
    current_time = datetime.now()
    if current_time < medication_reminder_time:
        speak(f"The medication is scheduled for 2 PM. Please ensure you take it then.")
    elif current_time >= medication_reminder_time:
        if not medication_taken:
            speak("It's after 2 PM. Have you taken your medication?")
            response = listen()
            if "yes" in response:
                medication_taken = True
                speak("Great! Remember to take it on time tomorrow as well.")
            elif "no" in response:
                speak("Please take your medication now. It's important for your health.")
            else:
                speak("I didn't catch that. Please let me know if you need help.")
        else:
            speak("You already took your medication. Good job!")

# Function to Check Medication Status
def check_medication_status():
    if medication_taken:
        speak("You already took your medication.")
    else:
        speak("You haven't taken your medication yet. It's important to take it on time.")

# Command-Specific Processing
def command_specific_processing(command):
    if "medication" in command:
        if "status" in command or "check" in command:
            check_medication_status()
        elif "time" in command or "reminder" in command:
            handle_medication_reminder()
        else:
            speak("I can help with medication status or reminders. Please clarify.")
    elif "exit" in command:
        speak("Goodbye!")
        return False
    else:
        speak("I'm not sure how to help with that. Try asking about medication.")
    return True

# NLP Processing for general queries
def nlp_processing(command):
    analysis = nlp(command)
    if "medication" in command:
        if "status" in analysis["label"]:
            check_medication_status()
        elif "reminder" in analysis["label"]:
            handle_medication_reminder()
        else:
            speak("Let me try to help with your medication query.")
    else:
        # Use GPT-2 for conversational responses
        response = conversation_model(command, max_length=50, truncation=True)
        speak(response[0]['generated_text'])
    return True

# Hybrid Command Processing
def process_command(command):
    if "medication" in command or "exit" in command:
        # Use command-specific processing for direct queries
        return command_specific_processing(command)
    else:
        # Use NLP for more flexible responses
        return nlp_processing(command)

# Continuous Listening Loop
def main():
    speak("System initialized. I'm listening for your commands.")
    running = True
    while running:
        print("Listening for command...")
        command = listen()
        if command:
            print(f"User: {command}")  # Debugging/logging
            running = process_command(command)

# Run the system
if __name__ == "__main__":
    main()
