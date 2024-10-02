import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.layers import Activation, Dense, LSTM, Input
import tkinter as tk
from tkinter import ttk
import html
import re
from datetime import datetime

# Decoding function for HTML entities
def decode_html_entities(text):
    return html.unescape(text)

# Function to clean the text data
def clean_text(text):
    # Remove any unwanted characters and HTML tags
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces/newlines with a single space
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'[^a-zA-Z0-9.,!?\'\"\- ]', '', text)  # Remove any non-alphanumeric characters except punctuation
    return text

# Load the dataset of formal letters or applications
filepath = tf.keras.utils.get_file(
    'formal_letters.txt', 
    'https://raw.githubusercontent.com/ayushmish2990/Mail-Format/main/Formal%20letter%20format.txt'
)
text = open(filepath, 'r', encoding='utf-8').read().lower()

# Clean the dataset text
text = clean_text(text)
# Use a portion of the text for training
text = text[:500000]  # Use the first 500,000 characters

# Character mapping
characters = sorted(set(text))
char_to_index = {c: i for i, c in enumerate(characters)}
index_to_char = {i: c for i, c in enumerate(characters)}

# Define sequence length and step size
SEQ_LENGTH = 40
STEP_SIZE = 3

# Prepare input sequences and corresponding next characters
sentences = []
next_char = []

for i in range(0, len(text) - SEQ_LENGTH, STEP_SIZE):
    sentences.append(text[i: i + SEQ_LENGTH])
    next_char.append(text[i + SEQ_LENGTH])

# One-hot encoding for input and output
x = np.zeros((len(sentences), SEQ_LENGTH, len(characters)), dtype=np.bool_)
y = np.zeros((len(sentences), len(characters)), dtype=np.bool_)

for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_to_index[char]] = True
    y[i, char_to_index[next_char[i]]] = True

# Build the model
model = Sequential()
model.add(Input(shape=(SEQ_LENGTH, len(characters))))  # Define the input layer explicitly
model.add(LSTM(128))
model.add(Dense(len(characters)))
model.add(Activation('softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer=RMSprop(learning_rate=0.01))

# Train the model
model.fit(x, y, batch_size=256, epochs=4)

# Sampling function
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# Define different fixed body templates
def get_fixed_body_template(purpose):
    if "leave" in purpose.lower():
        return (
            "\n\nI would like to kindly request a leave of absence from work "
            "for the following reasons. I will ensure that all my duties and "
            "responsibilities are properly handed over before my departure, "
            "and I will make sure to return by the specified date."
        )
    elif "complaint" in purpose.lower():
        return (
            "\n\nI am writing to formally lodge a complaint regarding the issue. "
            "I hope that you will look into this matter at the earliest and "
            "take the necessary actions to resolve the issue as soon as possible."
        )
    elif "enquiry" in purpose.lower():
        return (
            "\n\nThis is with reference to your advertisement in the ‘The Times of India’ for CAT Coaching classes."
            " I have passed the B.Sc. degree examination with Statistics as the main subject."
            "I am keen on joining your institute for the coaching classes."
            "Kindly let me know about the procedure of applying for the qualifying test and its date."
            "I would like to enroll as soon as possible. Your early response will enable me to decide fast."
        )
    elif "order" in purpose.lower():
        return (
            "\n\nThis is with reference to the Order No.(________) placed on Nov 17, 20xx."
            "The order consists of letterhead and business cards."
            "As per the agreement, we were promised to receive the order by Nov 22, 20xx."
            "The order did not reach on time, and the quality of the papers and design selected for business cards "
            "does not match the one selected. We faced a lot of embarrassment and inconvenience, and our reputation "
            "is at stake in the eyes of our clients. Kindly ensure that the order will be replaced by Dec 4, 20xx."
        )
    elif "promotion" in purpose.lower():
        return (
            "\n\nWe are glad to announce the grand opening of a new branch of our company in QPR Colony, Delhi on Dec 05, 20xx."
            "As a respected client, we are delighted to inform you that this branch offers various solutions to your problems."
            "We are dedicated to providing you with the best service and would be happy to have you as our guest."
        )
    elif "application" in purpose.lower():
        return (
            "\n\nI am submitting this application for the position that is available. "
            "I have attached my resume and relevant documents for your consideration. "
            "I am eager to discuss how my skills can contribute to your organization."
        )
    elif "invitation" in purpose.lower():
        return (
            "\n\nIt is with great pleasure that I invite you to the event being held on "
            "[date]. We would be honored to have your presence at this occasion, and I "
            "look forward to your confirmation of attendance."
        )
    return (
        "\n\nI am writing to you regarding the above-mentioned subject. "
        "I would like to formally request your attention to this matter and provide the necessary assistance. "
        "Your prompt response and cooperation would be highly appreciated."
    )

# Text generation function for letter/application format
def generate_text(length, temperature, recipient, sender, purpose):
    start_index = random.randint(0, len(text) - SEQ_LENGTH - 1)
    
    # Get the fixed body message based on the purpose
    fixed_body_message = get_fixed_body_template(purpose)
    
    # Generate the header with sender's address, date, and receiver's address
    date_today = datetime.now().strftime("%B %d, %Y")
    generated = (
        f"Sender's Address:\n[Your Address Here]\n\n"
        f"Date: {date_today}\n\n"
        f"Receiver's Address:\n{recipient}\n\n"
        f"Subject: {purpose}\n\n"
        f"Sir/Madam,\n\n"
    )
    
    sentence = text[start_index: start_index + SEQ_LENGTH]
    generated += sentence
    
    # Insert the fixed body message after the initial generated text
    generated += fixed_body_message
    
    # Generate additional text
    for _ in range(length):
        x_predictions = np.zeros((1, SEQ_LENGTH, len(characters)))
        for t, char in enumerate(sentence):
            x_predictions[0, t, char_to_index[char]] = 1

        predictions = model.predict(x_predictions, verbose=0)[0]
        next_index = sample(predictions, temperature)
        next_character = index_to_char[next_index]

        generated += next_character
        sentence = sentence[1:] + next_character
    
    # Add closing with sender's name
    generated += f"\n\nSincerely,\n{sender}"
    return generated

# Create the application window using Tkinter
def run_app():
    # Function to generate letter based on user input and display it
    def on_generate_text():
        try:
            length = int(length_entry.get())
            temperature = float(temp_entry.get())
            recipient = recipient_entry.get()
            sender = sender_entry.get()
            purpose = purpose_entry.get()
            generated_text = generate_text(length, temperature, recipient, sender, purpose)
            
            # Decode the generated text before displaying
            decoded_text = decode_html_entities(generated_text)
            
            output_text.delete('1.0', tk.END)  # Clear the previous output
            output_text.insert(tk.END, decoded_text)
        except ValueError:
            output_text.delete('1.0', tk.END)
            output_text.insert(tk.END, "Invalid input! Please enter valid numbers.")
    
    # Create the main window
    window = tk.Tk()
    window.title("Formal Letter/Application Generator")

    # Input fields for recipient, sender, and purpose
    recipient_label = ttk.Label(window, text="Recipient Name:")
    recipient_label.grid(column=0, row=0, padx=10, pady=10)
    
    recipient_entry = ttk.Entry(window)
    recipient_entry.grid(column=1, row=0, padx=10, pady=10)

    sender_label = ttk.Label(window, text="Sender Name:")
    sender_label.grid(column=0, row=1, padx=10, pady=10)
    
    sender_entry = ttk.Entry(window)
    sender_entry.grid(column=1, row=1, padx=10, pady=10)

    purpose_label = ttk.Label(window, text="Purpose of Letter:")
    purpose_label.grid(column=0, row=2, padx=10, pady=10)
    
    purpose_entry = ttk.Entry(window)
    purpose_entry.grid(column=1, row=2, padx=10, pady=10)

    # Input fields for text generation length and temperature
    length_label = ttk.Label(window, text="Length of Text:")
    length_label.grid(column=0, row=3, padx=10, pady=10)
    
    length_entry = ttk.Entry(window)
    length_entry.grid(column=1, row=3, padx=10, pady=10)
    length_entry.insert(0, '300')  # Default value

    temp_label = ttk.Label(window, text="Temperature (0.1 - 1.0):")
    temp_label.grid(column=0, row=4, padx=10, pady=10)
    
    temp_entry = ttk.Entry(window)
    temp_entry.grid(column=1, row=4, padx=10, pady=10)
    temp_entry.insert(0, '0.5')  # Default value

    # Button to generate text
    generate_button = ttk.Button(window, text="Generate Letter", command=on_generate_text)
    generate_button.grid(column=0, row=5, columnspan=2, padx=10, pady=10)

    # Output text box to display the generated text
    output_text = tk.Text(window, wrap='word', width=60, height=20)
    output_text.grid(column=0, row=6, columnspan=2, padx=10, pady=10)

    # Start the Tkinter event loop
    window.mainloop()

# Run the app
run_app()
