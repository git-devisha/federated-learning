import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
import flwr as fl
from collections import OrderedDict
from model import MentalHealthQAModel
from typing import Dict, List, Tuple

# Define the device for training and evaluation
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MentalHealthQADataset(Dataset):
    """
    Custom PyTorch Dataset for Mental Health MCQ Q&A.
    Prepares data for AutoModelForMultipleChoice.
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

        # Prepare inputs for AutoModelForMultipleChoice
        # Each choice forms a separate input sequence with context and question
        input_ids_choices = []
        attention_mask_choices = []
        for choice in choices:
            encoding = self.tokenizer(
                context,
                question + " " + choice, # Concatenate question and choice
                truncation=True,
                padding="max_length",
                max_length=128, # Keep sequence length reasonable for Q&A
                return_tensors="pt",
            )
            input_ids_choices.append(encoding["input_ids"])
            attention_mask_choices.append(encoding["attention_mask"])

        # Stack the choices' encodings to form a single input for the model
        # Shape: (1, num_choices, sequence_length)
        input_ids = torch.cat(input_ids_choices, dim=0).unsqueeze(0)
        attention_mask = torch.cat(attention_mask_choices, dim=0).unsqueeze(0)

        return {
            "input_ids": input_ids.squeeze(0), # Remove batch dim for DataLoader
            "attention_mask": attention_mask.squeeze(0), # Remove batch dim for DataLoader
            "labels": torch.tensor(correct_answer_index, dtype=torch.long)
        }

def load_simulated_client_data(client_id: int) -> Tuple[MentalHealthQADataset, MentalHealthQADataset]:
    """
    Loads a simulated local dataset for a given client.
    In a real application, this would load actual private data from the client's device.
    """
    # Simulate different data for different clients
    if client_id == 0:
        texts = [
            {
                "context": "A person feels persistently sad and loses interest in activities.",
                "question": "What mental health condition might this indicate?",
                "choices": ["Anxiety", "Depression", "Bipolar Disorder", "OCD"],
                "correct_answer_index": 1 # Depression
            },
            {
                "context": "Experiencing sudden, intense fear with physical symptoms like racing heart.",
                "question": "What is this commonly known as?",
                "choices": ["Panic attack", "Stress response", "Mild worry", "Excitement"],
                "correct_answer_index": 0 # Panic attack
            },
            {
                "context": "Difficulty sleeping, irritability, and excessive worry about everyday things.",
                "question": "These symptoms are often associated with:",
                "choices": ["Schizophrenia", "Generalized Anxiety Disorder", "ADHD", "Eating Disorder"],
                "correct_answer_index": 1 # Generalized Anxiety Disorder
            }
        ]
    elif client_id == 1:
        texts = [
            {
                "context": "Repetitive thoughts and compulsive behaviors.",
                "question": "Which disorder is characterized by these?",
                "choices": ["PTSD", "Bipolar Disorder", "Obsessive-Compulsive Disorder", "Social Anxiety"],
                "correct_answer_index": 2 # Obsessive-Compulsive Disorder
            },
            {
                "context": "A sudden and extreme shift in mood, energy, and activity levels.",
                "question": "This describes a key feature of:",
                "choices": ["Depression", "Anxiety", "Bipolar Disorder", "Phobia"],
                "correct_answer_index": 2 # Bipolar Disorder
            },
            {
                "context": "Feeling detached from reality, experiencing hallucinations or delusions.",
                "question": "These are symptoms often seen in:",
                "choices": ["Eating Disorders", "Schizophrenia", "Depression", "Panic Disorder"],
                "correct_answer_index": 1 # Schizophrenia
            }
        ]
    else: # For additional clients, if needed
        texts = [
            {
                "context": "Persistent difficulty concentrating and hyperactivity.",
                "question": "What condition might this suggest?",
                "choices": ["Autism", "ADHD", "Dyslexia", "Tourette's Syndrome"],
                "correct_answer_index": 1 # ADHD
            },
            {
                "context": "Intense fear of social situations, leading to avoidance.",
                "question": "This is characteristic of:",
                "choices": ["Generalized Anxiety", "Social Anxiety Disorder", "Agoraphobia", "Specific Phobia"],
                "correct_answer_index": 1 # Social Anxiety Disorder
            }
        ]

    # Split into train and test (simple split for simulation)
    train_size = int(0.8 * len(texts))
    train_data_points = texts[:train_size]
    test_data_points = texts[train_size:]

    train_dataset = MentalHealthQADataset(train_data_points)
    test_dataset = MentalHealthQADataset(test_data_points)

    return train_dataset, test_dataset

class MentalHealthQAClient(fl.client.NumPyClient):
    """
    Flower client for Mental Health Q&A.
    Performs local training and evaluation.
    """
    def __init__(self, client_id: int):
        self.client_id = client_id
        self.model = MentalHealthQAModel()
        self.model.to(DEVICE)
        self.train_dataset, self.test_dataset = load_simulated_client_data(client_id)
        self.train_loader = DataLoader(self.train_dataset, batch_size=1, shuffle=True) # Batch size 1 due to varying num_choices
        self.test_loader = DataLoader(self.test_dataset, batch_size=1)

    def get_parameters(self, config: Dict) -> List[fl.common.NDArray]:
        """Returns the current model parameters as a list of NumPy arrays."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: List[fl.common.NDArray]):
        """Sets the model parameters from a list of NumPy arrays."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters: List[fl.common.NDArray], config: Dict) -> Tuple[List[fl.common.NDArray], int, Dict]:
        """
        Trains the model locally on the client's data.
        """
        self.set_parameters(parameters) # Update model with global parameters

        # Get training configuration from server
        epochs = config.get("epochs", 1)
        learning_rate = config.get("learning_rate", 1e-5)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss() # For multiple choice classification

        self.model.train() # Set model to training mode
        print(f"Client {self.client_id} starting local training for {epochs} epochs...")
        for epoch in range(epochs):
            total_loss = 0.0
            for batch_idx, batch in enumerate(self.train_loader):
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                labels = batch["labels"].to(DEVICE)

                optimizer.zero_grad()
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss # AutoModelForMultipleChoice returns loss when labels are provided
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            print(f"Client {self.client_id} Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(self.train_loader):.4f}")

        print(f"Client {self.client_id} local training finished.")
        return self.get_parameters(config), len(self.train_dataset), {"loss": total_loss / len(self.train_loader)}

    def evaluate(self, parameters: List[fl.common.NDArray], config: Dict) -> Tuple[float, int, Dict]:
        """
        Evaluates the model locally on the client's test data and calculates MAG.
        """
        self.set_parameters(parameters) # Update model with global parameters

        self.model.eval() # Set model to evaluation mode
        correct_predictions = 0
        total_samples = 0
        total_loss = 0.0
        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for batch in self.test_loader:
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                labels = batch["labels"].to(DEVICE)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                logits = outputs.logits

                total_loss += loss.item()

                predicted_choice = torch.argmax(logits, dim=1)
                if predicted_choice.item() == labels.item():
                    correct_predictions += 1
                total_samples += 1

        mag = correct_predictions / total_samples if total_samples > 0 else 0.0
        avg_loss = total_loss / len(self.test_loader) if len(self.test_loader) > 0 else 0.0

        print(f"Client {self.client_id} Evaluation: Loss = {avg_loss:.4f}, Accuracy (MAG) = {mag*100:.2f}%")
        return avg_loss, len(self.test_dataset), {
    "mag": mag,
    "accuracy_percent": mag * 100
}


# Start client with .to_client() (Flower 2025 recommended way)
if __name__ == "__main__":
    # You can run multiple clients by changing the client_id
    # For example, run `python client.py 0` in one terminal and `python client.py 1` in another.
    import sys
    client_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    print(f"Starting Flower client {client_id}...")
    fl.client.start_client(
        server_address="127.0.0.1:8080",
        client=MentalHealthQAClient(client_id=client_id).to_client()
    )
    print(f"Flower client {client_id} stopped.")

