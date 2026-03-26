import torch

print("Training with fairness loss...")

# Dummy loss values
original_loss = 0.5
fairness_penalty = 0.1

total_loss = original_loss + fairness_penalty

print("Total Loss:", total_loss)