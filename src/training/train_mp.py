from typing import List
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset
import wandb

from animatediff.dataset import make_dataloader, make_dataset

def train_mp(
    model: nn.Module,
    dataset: Dataset,
    num_epochs: int = 10,
    batch_size: int = 4,
    lr: float = 0.001,
    gradient_accumulation_steps: int = 4,
    clip_grad_norm: float = 1.0,
    shards: List[str],
    wandb_project: str,
    wandb_entity: str,
    device_id: int
):
    """
    Train a model using PyTorch DistributedDataParallel.

    Args:
        model (nn.Module): The PyTorch model to train.
        dataset (Dataset): Dataset object to use for training.
        num_epochs (int, optional): Total number of epochs to train. Defaults to 10.
        batch_size (int, optional): Batch size per process. Defaults to 4.
        lr (float, optional): Learning rate for the optimizer. Defaults to 0.001.
        gradient_accumulation_steps (int, optional): Number of steps to accumulate gradients before a backward pass. Defaults to 4.
        clip_grad_norm (float, optional): Max norm for gradient clipping. Defaults to 1.0.
        shards (List[str]): Paths to dataset shards.
        wandb_project (str): Project name for Weights & Biases logging.
        wandb_entity (str): Entity name for Weights & Biases.
        device_id (int): GPU device ID for this process.
    """
    # Initialize WandB
    wandb.init(project=wandb_project, entity=wandb_entity)

    # Set up the device and DDP
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model = DDP(model, device_ids=[device_id], output_device=device_id)

    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Prepare the data loader
    dataset = make_dataset(shards, cache_dir="./tmp")
    dataloader = make_dataloader(dataset, batch_size=batch_size)

    # Training loop
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False)
        for step, batch in progress_bar:
            start_tokens, end_tokens = batch['start_tokens'].to(device), batch['end_tokens'].to(device)
            ground_truth = batch['ground_truth'].to(device)

            # Forward pass
            outputs = model(start_tokens, end_tokens)
            loss = criterion(outputs, ground_truth) / gradient_accumulation_steps

            # Backward pass
            loss.backward()

            if (step + 1) % gradient_accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)

                # Optimizer step
                optimizer.step()
                optimizer.zero_grad()

                # Log to WandB
                wandb.log({"train_loss": loss.item() * gradient_accumulation_steps, "epoch": epoch})
                epoch_loss += loss.item() * gradient_accumulation_steps

            # Update the progress bar
            progress_bar.set_postfix(loss=epoch_loss / (step + 1))

        # Save checkpoint
        if dist.get_rank() == 0:
            torch.save(model.state_dict(), f"checkpoint_epoch_{epoch}.pth")
            wandb.save(f"checkpoint_epoch_{epoch}.pth")

    # Cleanup
    dist.barrier()
    wandb.finish()
