import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from model import MentalHealthQAModel
from typing import List, Dict, Tuple
import random
import numpy as np
from tqdm import tqdm
import os

# Set seeds for reproducibility
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

# Define the device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MentalHealthQADataset(torch.utils.data.Dataset):
    """
    Custom PyTorch Dataset for Mental Health MCQ Q&A.
    """
    def __init__(self, data_points: List[Dict]):
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.data_points = data_points

    def __len__(self):
        return len(self.data_points)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data_points[idx]
        context = item["context"]
        question = item["question"]
        choices = item["choices"]
        correct_answer_index = item["correct_answer_index"]

        input_ids_choices = []
        attention_mask_choices = []
        for choice in choices:
            encoding = self.tokenizer(
                context,
                question + " " + choice,
                truncation=True,
                padding="max_length",
                max_length=128,
                return_tensors="pt",
            )
            input_ids_choices.append(encoding["input_ids"])
            attention_mask_choices.append(encoding["attention_mask"])

        input_ids = torch.cat(input_ids_choices, dim=0)
        attention_mask = torch.cat(attention_mask_choices, dim=0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(correct_answer_index, dtype=torch.long)
        }


def load_centralized_full_dataset() -> Tuple[MentalHealthQADataset, MentalHealthQADataset]:
    """
    Loads the combined dataset for centralized training and evaluation.
    """
    all_texts = []

    client_0_data = [
        {"context": "A person feels persistently sad and loses interest in activities.",
         "question": "What mental health condition might this indicate?",
         "choices": ["Anxiety", "Depression", "Bipolar Disorder", "OCD"], "correct_answer_index": 1},
        {"context": "Experiencing sudden, intense fear with physical symptoms like racing heart.",
         "question": "What is this commonly known as?",
         "choices": ["Panic attack", "Stress response", "Mild worry", "Excitement"], "correct_answer_index": 0},
        {"context": "Difficulty sleeping, irritability, and excessive worry about everyday things.",
         "question": "These symptoms are often associated with:",
         "choices": ["Schizophrenia", "Generalized Anxiety Disorder", "ADHD", "Eating Disorder"], "correct_answer_index": 1}
    ]

    client_1_data = [
        {"context": "Repetitive thoughts and compulsive behaviors.",
         "question": "Which disorder is characterized by these?",
         "choices": ["PTSD", "Bipolar Disorder", "Obsessive-Compulsive Disorder", "Social Anxiety"], "correct_answer_index": 2},
        {"context": "A sudden and extreme shift in mood, energy, and activity levels.",
         "question": "This describes a key feature of:",
         "choices": ["Depression", "Anxiety", "Bipolar Disorder", "Phobia"], "correct_answer_index": 2},
        {"context": "Feeling detached from reality, experiencing hallucinations or delusions.",
         "question": "These are symptoms often seen in:",
         "choices": ["Eating Disorders", "Schizophrenia", "Depression", "Panic Disorder"], "correct_answer_index": 1}
    ]

    additional_data = [
        {"context": "Persistent difficulty concentrating and hyperactivity.",
         "question": "What condition might this suggest?",
         "choices": ["Autism", "ADHD", "Dyslexia", "Tourette's Syndrome"], "correct_answer_index": 1},
        {"context": "Intense fear of social situations, leading to avoidance.",
         "question": "This is characteristic of:",
         "choices": ["Generalized Anxiety", "Social Anxiety Disorder", "Agoraphobia", "Specific Phobia"], "correct_answer_index": 1},
        {"context": "Recurrent binge eating followed by compensatory behaviors.",
         "question": "This is a symptom of which eating disorder?",
         "choices": ["Anorexia Nervosa", "Bulimia Nervosa", "Binge Eating Disorder", "Orthorexia"], "correct_answer_index": 1}
    ]

    all_texts.extend(client_0_data + client_1_data + additional_data)

    random.shuffle(all_texts)
    train_size = int(0.8 * len(all_texts))
    train_data_points = all_texts[:train_size]
    test_data_points = all_texts[train_size:]

    train_dataset = MentalHealthQADataset(train_data_points)
    test_dataset = MentalHealthQADataset(test_data_points)

    return train_dataset, test_dataset


def calculate_mag(model: torch.nn.Module, data_loader: DataLoader) -> float:
    """
    Calculates Mean Average Grade (MAG) - equivalent to accuracy.
    """
    model.eval()
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            predicted_choice = torch.argmax(logits, dim=1)
            correct_predictions += (predicted_choice == labels).sum().item()
            total_samples += labels.size(0)

    return correct_predictions / total_samples if total_samples > 0 else 0.0


if __name__ == "__main__":
    print("Starting improved centralized training...")

    # Load data
    train_dataset, test_dataset = load_centralized_full_dataset()
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=2)

    # Initialize model
    model = MentalHealthQAModel()
    model.to(DEVICE)

    # Optimizer & Scheduler
    learning_rate = 2e-5
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    num_epochs = 20
    total_steps = len(train_loader) * num_epochs
    warmup_steps = int(0.1 * total_steps)

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    best_mag = 0.0
    patience = 3
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in loop:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} finished. Average Loss: {avg_loss:.4f}")

        # Evaluate after each epoch
        mag_score = calculate_mag(model, test_loader)
        print(f"Validation MAG (Accuracy): {mag_score:.4f}")

        # Early Stopping
        if mag_score > best_mag:
            best_mag = mag_score
            patience_counter = 0
            torch.save(model.state_dict(), "best_centralized_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    # Load best model for final evaluation
    print("\nLoading best model for final evaluation...")
    model.load_state_dict(torch.load("best_centralized_model.pth"))
    final_mag = calculate_mag(model, test_loader)
    print(f"Best Centralized Model MAG (Accuracy): {final_mag*100:.2f}%")
