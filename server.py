import flwr as fl
import torch
from collections import OrderedDict
from typing import Dict, List, Tuple, Optional
from model import MentalHealthQAModel
from transformers import AutoTokenizer
from flwr.common import ndarrays_to_parameters

# Define the device for evaluation
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_centralized_test_dataset(num_clients: int = 2) -> List[Dict]:
    """
    Simulates a centralized test dataset for server-side evaluation.
    """
    all_texts = []

    for i in range(num_clients):
        texts_client = [
            {
                "context": f"Client {i} has been feeling overwhelmed with work and personal life.",
                "question": "What is a common sign of being overwhelmed?",
                "choices": ["Increased focus", "Feeling irritable", "More social interaction", "Better sleep"],
                "correct_answer_index": 1
            },
            {
                "context": f"Client {i} is trying to cope with anxiety.",
                "question": "Which coping mechanism is generally recommended?",
                "choices": ["Avoiding triggers", "Deep breathing exercises", "Ignoring feelings", "Excessive caffeine intake"],
                "correct_answer_index": 1
            },
            {
                "context": f"Client {i} is experiencing symptoms of low mood.",
                "question": "What activity can often help improve mood?",
                "choices": ["Staying indoors all day", "Engaging in physical activity", "Isolating from friends", "Overthinking problems"],
                "correct_answer_index": 1
            }
        ]
        all_texts.extend(texts_client)

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    tokenized_data = []
    for item in all_texts:
        context = item["context"]
        question = item["question"]
        choices = item["choices"]
        correct_answer_index = item["correct_answer_index"]

        input_ids_choices = []
        attention_mask_choices = []
        for choice in choices:
            encoding = tokenizer(
                context,
                question + " " + choice,
                truncation=True,
                padding="max_length",
                max_length=128,
                return_tensors="pt",
            )
            input_ids_choices.append(encoding["input_ids"])
            attention_mask_choices.append(encoding["attention_mask"])

        input_ids = torch.cat(input_ids_choices, dim=0).unsqueeze(0)
        attention_mask = torch.cat(attention_mask_choices, dim=0).unsqueeze(0)

        tokenized_data.append({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(correct_answer_index, dtype=torch.long).unsqueeze(0)
        })
    return tokenized_data

def calculate_mag(model: torch.nn.Module, data: List[Dict]) -> float:
    """
    Calculates the Mean Average Grade (MAG).
    """
    model.eval()
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for item in data:
            input_ids = item["input_ids"].to(DEVICE)
            attention_mask = item["attention_mask"].to(DEVICE)
            labels = item["labels"].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            predicted_choice = torch.argmax(logits, dim=1)

            if predicted_choice.item() == labels.item():
                correct_predictions += 1
            total_samples += 1

    return correct_predictions / total_samples if total_samples > 0 else 0.0

# Load centralized test data
centralized_test_data = load_centralized_test_dataset(num_clients=2)

class FedAvgMAG(fl.server.strategy.FedAvg):
    """
    Federated Averaging with server-side MAG evaluation.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.model = MentalHealthQAModel()
        self.model.to(DEVICE)

    def configure_evaluate(
        self, server_round: int, parameters: fl.common.Parameters, client_manager: fl.server.ClientManager
    ) -> List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateIns]]:
        if self.evaluate_fn:
            return []
        return super().configure_evaluate(server_round, parameters, client_manager)

from flwr.common import parameters_to_ndarrays

def evaluate(
    self, server_round: int, parameters: fl.common.Parameters
) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
    if self.evaluate_fn:
        return self.evaluate_fn(server_round, parameters, {})

    # Convert Flower Parameters to state_dict
    ndarrays = parameters_to_ndarrays(parameters)
    state_dict = OrderedDict({
        k: torch.tensor(v) for k, v in zip(self.model.state_dict().keys(), ndarrays)
    })
    self.model.load_state_dict(state_dict)

    # Evaluate
    mag = calculate_mag(self.model, centralized_test_data)
    print(f"Server-side evaluation round {server_round}: MAG = {mag:.4f}")

    return 0.0, {"mag": mag}

if __name__ == "__main__":
    initial_model = MentalHealthQAModel()
    model_weights = [val.cpu().numpy() for _, val in initial_model.state_dict().items()]
    initial_parameters = ndarrays_to_parameters(model_weights)

    strategy = FedAvgMAG(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_available_clients=2,
        min_fit_clients=2,
        min_evaluate_clients=2,
        initial_parameters=initial_parameters,
        evaluate_fn=None
    )

    print("Starting Flower server...")
    fl.server.start_server(
        server_address="127.0.0.1:8080",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
    )
    print("Flower server stopped.")
