"""
Train an EEGLLM to predict to Ojas's hand movements from his brain signals.
Implement all previous techniques like VQVAE tokenization on appendage vector
components and LaBraM.
"""

import yaml
import torch
import wandb
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from eeg.eeg_data import EEGDataset, EEGLLM

with open("config/eeg_llm.yaml", "r") as config_file:
    config = yaml.safe_load(config_file)

    vocab_size = config["vocab_size"]
    num_layers = config["num_layers"]
    num_heads = config["num_heads"]
    num_channels = config["num_channels"]
    embedding_dim = config["embedding_dim"]
    ffn_hidden_dim = config["ffn_hidden_dim"]
    qk_length = config["qk_length"]
    value_length = config["value_length"]
    max_length = config["max_length"]

    num_channels = config["num_channels"]
    num_times = config["num_times"]
    num_outputs = config["num_outputs"]

    dropout = config["dropout"]

    device = config["device"]
    batch_size = config["batch_size"]
    warmup_steps = config["warmup_steps"]
    base_lr = float(config["base_lr"])
    epochs = config["epochs"]

    run_name = config["run_name"]
    use_ckpt_path = config["use_ckpt_path"]
    save_ckpt_path = config["save_ckpt_path"]
    save_every = config["save_every"]

hand_dataset: EEGDataset = EEGDataset()
hand_dataloader = DataLoader(hand_dataset, batch_size=32, shuffle=True)

model = EEGLLM(
    vocab_size=vocab_size,
    num_layers=num_layers,
    num_heads=num_heads,
    num_channels=num_channels,
    embedding_dim=embedding_dim,
    ffn_hidden_dim=ffn_hidden_dim,
    qk_length=qk_length,
    value_length=value_length,
    max_length=max_length,
    dropout=dropout
)

optimizer = AdamW(model.parameters(), lr=base_lr, betas=(0.9, 0.98), eps=1e-9)
loss_fn = CrossEntropyLoss()

model.to(device)

def train():
    run = wandb.init(
        name=run_name,
        entity="prabhuneojas-evergreen-valley-high-school",
        project="eeg",
        config={
            "learning_rate": base_lr,
            "architecture": "Transformer",
            "dataset": "eeg_dataset",
            "epochs": epochs,
        },
    )

    model.to(device)

    epoch_tqdm = tqdm(range(epochs), dynamic_ncols=True)
    for i in epoch_tqdm:
        epoch_tqdm.set_description(f"Epoch {i + 1}")


        iter_tqdm = tqdm(hand_dataloader, dynamic_ncols=True)
        for eeg, apps, tokens in iter_tqdm:
            # chunk: (B, T, C)

            eeg = eeg.to(device).float()
            apps = apps.to(device).float()
            tokens = tokens.to(device)

            token_logits = model(eeg) # out: (B, T, vocab_size)
            token_logits = token_logits.transpose(1, 2) # (B, T, vocab_size) -> (B, vocab_size, T)

            loss = loss_fn(token_logits, tokens)

            iter_tqdm.set_postfix({"loss": loss.item()})
            run.log({"loss": loss.item()})

            optimizer.zero_grad()  # optimizer has access to all model params, makes grads 0
            loss.backward()  # calculates and adds gradients to params so optim sees
            optimizer.step()  # optim looks at gradients and steps accordingly

        # if (i + 1) % save_every == 0:
        #     latest_ckpt = {
        #         "epochs": i,
        #         "model": model.state_dict(),
        #         "optimizer": optimizer.state_dict(),
        #     }

        #     torch.save(latest_ckpt, f"{save_ckpt_path}_epoch_{i + 1}.pth")

    run.finish()


train()

# latest_ckpt = {
#     "epochs": epochs,
#     "model": model.state_dict(),
#     "optimizer": optimizer.state_dict(),
# }

# torch.save(latest_ckpt, save_ckpt_path)
