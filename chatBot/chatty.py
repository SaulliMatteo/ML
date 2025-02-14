from transformers import pipeline

chatty = pipeline("text-generation", model="gpt2")
print("Benvento/a nel Chatbot! Digita 'quit' per uscire.")

while True:
    userInput = input("You: ")
    if userInput.lower() == 'quit':
        break

    response = chatty(userInput, max_length=100, num_return_sequences=1)
    print("chatty: ", response[0]["generated_text"])
