import pandas as pd
import torch
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def load_and_split_data(data_path, text_column, label_column, test_size=0.2, random_state=42):
    """Load data from a CSV file and split it into train and validation sets."""
    data = pd.read_csv(data_path)
    texts = data[text_column].tolist()
    labels = data[label_column].tolist()
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=test_size, random_state=random_state
    )
    return train_texts, val_texts, train_labels, val_labels


def normalize_labels(labels):
    """Convert labels to a consistent type and return numeric labels with a label map."""
    labels = [str(label) for label in labels]  # Convert to strings
    label_map = {label: idx for idx, label in enumerate(sorted(set(labels)))}
    numeric_labels = [label_map[label] for label in labels]
    return numeric_labels, label_map


def create_dataloader(texts, labels, tokenizer, batch_size=16):
    """Tokenize texts and create a DataLoader."""
    encoded_dict = tokenizer.batch_encode_plus(
        texts,
        add_special_tokens=True,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    input_ids = encoded_dict['input_ids']
    attention_masks = encoded_dict['attention_mask']

    # Convert labels to tensors
    labels = torch.tensor(labels)

    # Create TensorDataset and DataLoader
    dataset = TensorDataset(input_ids, attention_masks, labels)
    sampler = RandomSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
    return dataloader


def initialize_model(num_labels, learning_rate=2e-5):
    """Initialize the BERT model for sequence classification."""
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=num_labels
    )
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    return model, optimizer


def train_model(model, train_dataloader, val_dataloader, optimizer, epochs=3):
    """Train the BERT model and evaluate on validation set."""
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    for epoch in range(epochs):
        # Training loop
        print(f"Epoch {epoch + 1}/{epochs}: Training...")
        model.train()
        for batch in train_dataloader:
            # Move batch to device
            batch = tuple(t.to(device) for t in batch)
            input_ids, attention_masks, labels = batch

            # Forward pass
            outputs = model(input_ids, attention_mask=attention_masks, labels=labels)
            loss = outputs.loss

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Evaluation loop
        print("Evaluating...")
        model.eval()
        predictions, true_labels = [], []
        for batch in val_dataloader:
            batch = tuple(t.to(device) for t in batch)
            input_ids, attention_masks, labels = batch

            with torch.no_grad():
                outputs = model(input_ids, attention_mask=attention_masks)
                logits = outputs.logits
                predictions.extend(torch.argmax(logits, axis=1).tolist())
                true_labels.extend(labels.tolist())

        print(classification_report(true_labels, predictions))


def main(data_path, save_path, text_column, label_column):
    """Main function to train and save the BERT model."""
    # Step 1: Load and split data
    train_texts, val_texts, train_labels, val_labels = load_and_split_data(
        data_path, text_column, label_column
    )

    # Step 2: Normalize labels
    train_labels, label_map = normalize_labels(train_labels)
    val_labels, _ = normalize_labels(val_labels)  # Use the same mapping for validation

    # Step 3: Initialize tokenizer and DataLoaders
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    train_dataloader = create_dataloader(train_texts, train_labels, tokenizer)
    val_dataloader = create_dataloader(val_texts, val_labels, tokenizer)

    # Step 4: Initialize model and optimizer
    num_labels = len(label_map)
    model, optimizer = initialize_model(num_labels)

    # Step 5: Train the model
    train_model(model, train_dataloader, val_dataloader, optimizer)

    # Step 6: Save the trained model
    model.save_pretrained(save_path)
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    main(
        data_path='Data/final_courses_with_paths2.csv',
        save_path='data/saved_model_directory',
        text_column='short_description',
        label_column='Path_LLM'
    )
