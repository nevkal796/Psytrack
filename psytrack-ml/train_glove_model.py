# 1. Imports
import os
import re
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2

try:
    from colorama import Fore, Style
except ImportError:
    Fore = Style = None

# 2. Load dataset
df = pd.read_csv("mental_health.csv")
print("🚀 Script started...")

# 3. Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df['clean_text'] = df['text'].astype(str).apply(clean_text)

# 4. Tokenization
vocab_size = 10000
max_length = 100
oov_token = "<OOV>"

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
tokenizer.fit_on_texts(df['clean_text'])
word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(df['clean_text'])
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')
labels = np.array(df['label'])

# 5. Load GloVe embeddings
print("📦 Loading GloVe vectors...")
embedding_index = {}
with open("glove.6B.100d.txt", encoding="utf8") as f:
    for line in f:
        values = line.strip().split()
        word = values[0]
        coeffs = np.asarray(values[1:], dtype='float32')
        embedding_index[word] = coeffs

# 6. Create embedding matrix
embedding_dim = 100
embedding_matrix = np.zeros((vocab_size, embedding_dim))

for word, i in word_index.items():
    if i < vocab_size:
        vector = embedding_index.get(word)
        if vector is not None:
            embedding_matrix[i] = vector

# 7. Define model using pre-trained embeddings
def create_model():
    model = Sequential([
        Embedding(input_dim=vocab_size,
                  output_dim=embedding_dim,
                  weights=[embedding_matrix],
                  input_length=max_length,
                  trainable=False),  # keep GloVe frozen
        GlobalAveragePooling1D(),
        Dense(24, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# 8. Train or load model
model_path = "mental_health_glove.keras"
if os.path.exists(model_path):
    print("📥 Loading saved model...")
    model = load_model(model_path)
else:
    print("🧠 Training model with GloVe embeddings...")
    model = create_model()
    early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
    model.fit(padded_sequences, labels, epochs=20, validation_split=0.2, callbacks=[early_stop])
    model.save(model_path)
    print("✅ Model trained and saved.")

# 9. Input processing
def prepare_input(text):
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=max_length, padding='post', truncating='post')
    return padded

# 10. Feedback generation
def get_feedback(text):
    input_seq = prepare_input(text)
    prediction = model.predict(input_seq)[0][0]

    if prediction > 0.5:
        feedback = f"\n🟥 Negative tone detected ({prediction:.2f}).\nTry reflecting, resting, or reaching out to someone if needed."
        return f"{Fore.RED}{feedback}{Style.RESET_ALL}" if Fore else feedback
    else:
        feedback = f"\n🟩 Positive tone detected ({1 - prediction:.2f}).\nGlad you're feeling good — keep journaling!"
        return f"{Fore.GREEN}{feedback}{Style.RESET_ALL}" if Fore else feedback

# 11. Run journaling loop
print("\n🤖 Model is ready.")
print("Type your journal entry (or 'quit' to exit):\n")

while True:
    try:
        entry = input("📝 Your journal entry: ").strip()
        if entry.lower() == 'quit':
            print("👋 Exiting. Take care!")
            break
        if not entry:
            print("⚠️ Please enter some text.")
            continue
        print(get_feedback(entry))
    except KeyboardInterrupt:
        print("\n👋 Exiting due to keyboard interrupt.")
        break
