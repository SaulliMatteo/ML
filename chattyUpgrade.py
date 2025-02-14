
from transformers import pipeline
from deep_translator import GoogleTranslator

traslatorITEN = GoogleTranslator(sourch = "it", target = "en")
traslatorENIT = GoogleTranslator(sourch = "en", target = "it")

chatty = pipeline("text-generation", model="gpt2")
print("Benvento/a nel Chatbot! Digita 'quit' per uscire.")

while True:
    userInput = input("You: ")
    if userInput.lower() == 'quit':
        break

    userInputEN = traslatorITEN.translate(userInput)

    response = chatty(userInput, max_length=100, num_return_sequences=1, temperature = 0.8, top_k = 5, top_p = 0.7)
    print("chatty: ", traslatorENIT.translate(response[0]["generated_text"]))