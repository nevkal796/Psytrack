import pandas as pd
from datasets import Dataset
from transformers import DistilBertTokenizerFast, TFDistilBertForSequenceClassification, DefaultDataCollator
import tensorflow as tf

# 1. Load CSV with text and label columns
df = pd.read_csv("mental_health.csv")  # Make sure it has 'text' and 'label' columns

# 2. Convert pandas DataFrame to Hugging Face Dataset
dataset = Dataset.from_pandas(df)

# 3. Load tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

# 4. Tokenize dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

dataset = dataset.map(tokenize_function, batched=True)

# 5. Split dataset into train/test
split = dataset.train_test_split(test_size=0.2)

# 6. Prepare TensorFlow datasets, specifying label_cols
train_ds = split["train"].to_tf_dataset(
    columns=["input_ids", "attention_mask"],
    label_cols=["label"],
    shuffle=True,
    batch_size=32,
    collate_fn=DefaultDataCollator(return_tensors="tf"),
)

val_ds = split["test"].to_tf_dataset(
    columns=["input_ids", "attention_mask"],
    label_cols=["label"],
    shuffle=False,
    batch_size=32,
    collate_fn=DefaultDataCollator(return_tensors="tf"),
)

# 7. Load model
model = TFDistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")

# 8. Compile model
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]

model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

# 9. Train model
model.fit(train_ds, validation_data=val_ds, epochs=3)

# 10. Save model
model.save_pretrained("./fine_tuned_distilbert_model")
tokenizer.save_pretrained("./fine_tuned_distilbert_model")
