import torch
import torch.nn as nn
from transformers import AutoModelForMultipleChoice, AutoTokenizer

class MentalHealthQAModel(nn.Module):
    """
    A PyTorch model for Multiple Choice Question Answering in a mental health context.
    It leverages a pre-trained BERT-based model (bert-base-uncased) and adapts it
    for multiple-choice classification using AutoModelForMultipleChoice.
    """
    def __init__(self):
        super().__init__()
        # Load a pre-trained model suitable for multiple choice tasks.
        # AutoModelForMultipleChoice automatically adds a classification head
        # on top of the base model (e.g., BERT), which outputs logits for each choice.
        self.model = AutoModelForMultipleChoice.from_pretrained('bert-base-uncased')

    def forward(self, input_ids, attention_mask, labels=None):
        """
        Forward pass for the Q&A model.

        Args:
            input_ids (torch.Tensor): Tensor of input token IDs.
                                      Shape: (batch_size, num_choices, sequence_length)
            attention_mask (torch.Tensor): Tensor of attention masks.
                                           Shape: (batch_size, num_choices, sequence_length)
            labels (torch.Tensor, optional): Tensor of correct choice indices for training.
                                             Shape: (batch_size,)

        Returns:
            transformers.modeling_outputs.MultipleChoiceModelOutput:
                A tuple containing logits (if labels is None) or a loss and logits (if labels is provided).
                Logits shape: (batch_size, num_choices)
        """
        # The AutoModelForMultipleChoice expects input_ids and attention_mask
        # to have a shape of (batch_size, num_choices, sequence_length).
        # It then flattens the first two dimensions internally for processing.
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels # Pass labels directly for loss calculation during training
        )
        return outputs

