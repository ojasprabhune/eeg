"""
Experiment scheduler: train the language model per experiment and per input
accuracy, measure character error rate (CER) with autoregressive decoding, and
save plots.

For each experiment (asl_8_letters, common_8_letters, 6_letters) the right
gesture dictionary is chosen automatically (via gesture_experiments) and the
right corpus is loaded (via LanguageDataset, which reads <experiment>_corpus.txt).
For each input-sequence accuracy in ACCURACIES we build the dataset at that
accuracy, train a fresh model, greedily decode the validation set, and compute
mean CER.

Outputs:
  - ckpts/lm/<experiment>_acc<acc>.pth        (one model per run)
  - figures/cer_vs_accuracy.png               (line per experiment)
  - figures/cer_vs_experiment.png             (bar at clean accuracy = 1.0)
  - figures/results.json                      (raw numbers)

The model is tiny (<1M params), so this runs fine on CPU. This does NOT use
wandb and does NOT load any large model, so it is safe to run in one process.
"""

import json
import os

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import torch
import yaml
from torch.nn import CrossEntropyLoss, MSELoss
from torch.optim import AdamW
from torch.utils.data import DataLoader

from eeg.language_model import LanguageDataset, LanguageModel
from eeg.language_model.metrics import compute_cer
from eeg.language_model.tokenizer import LanguageTokenizer

# --- configuration -----------------------------------------------------------

with open("config/language_model.yaml", "r") as f:
    CONFIG = yaml.safe_load(f)

EXPERIMENTS = ["asl_8_letters", "common_8_letters", "6_letters"]
ACCURACIES = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

EPOCHS = int(os.environ.get("SWEEP_EPOCHS", 400))  # per (experiment, accuracy) run
# larger batch => far fewer steps/epoch; this model is tiny and overhead-bound,
# so a big batch is dramatically faster (esp. on GPU) for the same convergence
BATCH_SIZE = int(os.environ.get("SWEEP_BATCH", 32))
DEVICE = os.environ.get("SWEEP_DEVICE", "cpu")
# scheduled-sampling k: smaller => epsilon decays faster => the model trains on
# its own predictions sooner, so autoregressive CER converges in fewer epochs.
# The training config uses 55555 (tuned for a 150k-epoch run); for this sweep we
# ramp much faster.
K = int(os.environ.get("SWEEP_K", CONFIG["k"]))

CKPT_DIR = os.environ.get("SWEEP_CKPT_DIR", "ckpts/lm")
FIG_DIR = os.environ.get("SWEEP_FIG_DIR", "figures")

SOS, EOS, PAD = 1, 2, 0  # from LanguageTokenizer


def num_classes_for(experiment: str) -> int:
    return 6 if experiment == "6_letters" else 4


# --- decoding + CER ----------------------------------------------------------


@torch.no_grad()
def greedy_decode(model, features, feature_masks, max_len, device):
    """Autoregressively decode letter tokens from encoder features."""
    model.eval()
    B = features.size(0)
    src = features.to(device)
    src_pad = ~feature_masks.to(device).bool()

    ys = torch.full((B, 1), SOS, dtype=torch.long, device=device)
    finished = torch.zeros(B, dtype=torch.bool, device=device)

    for _ in range(max_len):
        tgt_pad = torch.zeros_like(ys, dtype=torch.bool)
        logits, _ = model(
            src=src,
            tgt=ys,
            src_pad_mask=src_pad,
            tgt_pad_mask=tgt_pad,
            step=0,
            return_epsilon=False,
            use_scheduled_sampling=False,
        )
        next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # (B, 1)
        ys = torch.cat([ys, next_tok], dim=1)
        finished |= next_tok.squeeze(1) == EOS
        if finished.all():
            break

    return ys


def evaluate_cer(model, dataset, tokenizer, device) -> float:
    """Mean CER over the validation split using greedy autoregressive decoding."""
    features = torch.tensor(dataset.val_features)
    feature_masks = torch.tensor(dataset.val_feature_masks)
    labels = torch.tensor(dataset.val_labels)

    if features.size(0) == 0:
        return float("nan")

    max_len = labels.size(1) + 2
    preds = greedy_decode(model, features, feature_masks, max_len, device)

    # greedy_decode keeps stepping every row until ALL rows emit EOS, so a row
    # that finished early has garbage letters appended after its EOS. The
    # tokenizer only strips SOS/EOS/PAD, not "everything after the first EOS",
    # so blank those trailing tokens out before decoding (otherwise even the
    # identity case never reaches CER 0).
    preds = preds.clone()
    for r in range(preds.size(0)):
        eos_pos = (preds[r] == EOS).nonzero()
        if len(eos_pos) > 0:
            preds[r, eos_pos[0].item() :] = PAD

    pred_strs = tokenizer.decode(preds.cpu())
    true_strs = tokenizer.decode(labels.cpu())

    cers = [compute_cer(p, t) for p, t in zip(pred_strs, true_strs)]
    return sum(cers) / len(cers)


# --- training ----------------------------------------------------------------


def train_one(experiment: str, accuracy: float) -> float:
    """Train a fresh model at the given accuracy and return validation CER."""
    num_classes = num_classes_for(experiment)

    train_ds = LanguageDataset(
        experiment=experiment,
        num_classes=num_classes,
        mode="train",
        device=DEVICE,
        accuracy=accuracy,
    )
    val_ds = LanguageDataset(
        experiment=experiment,
        num_classes=num_classes,
        mode="val",
        device=DEVICE,
        accuracy=accuracy,
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    model = LanguageModel(
        vocab_size=CONFIG["vocab_size"],
        num_layers=CONFIG["num_layers"],
        decoder_num_layers=CONFIG["decoder_num_layers"],
        num_heads=CONFIG["num_heads"],
        num_inputs_classes=num_classes,
        decoder_embedding_dim=CONFIG["decoder_embedding_dim"],
        ffn_hidden_dim=CONFIG["ffn_hidden_dim"],
        encoder_dropout=CONFIG["encoder_dropout"],
        decoder_dropout=CONFIG["decoder_dropout"],
        k=K,
        min_value=CONFIG["min_value"],
    ).to(DEVICE)

    optimizer = AdamW(
        model.parameters(), lr=float(CONFIG["base_lr"]), betas=(0.9, 0.98), eps=1e-9
    )
    ce_loss_fn = CrossEntropyLoss(ignore_index=PAD)
    mse_loss_fn = MSELoss()
    recon_lambda = CONFIG["recon_lambda"]

    step = 0
    model.train()
    for _ in range(EPOCHS):
        for feature, feature_mask, label, label_mask, _ in train_loader:
            feature = feature.to(DEVICE)
            feature_mask = feature_mask.to(DEVICE).bool()
            label = label.to(DEVICE).to(torch.int64)
            label_mask = label_mask.to(DEVICE).bool()

            in_feature, gt_feature = feature[:, :-1, :], feature[:, 1:, :]
            in_feature_mask = feature_mask[:, :-1]
            in_label, gt_label = label[:, :-1], label[:, 1:]
            in_label_mask = label_mask[:, :-1]

            logits, recon, _ = model(
                src=in_feature,
                tgt=in_label,
                src_pad_mask=~in_feature_mask,
                tgt_pad_mask=~in_label_mask,
                step=step,
                return_epsilon=True,
            )
            logits = logits.transpose(1, 2)
            loss = ce_loss_fn(logits, gt_label) + recon_lambda * mse_loss_fn(
                recon, gt_feature
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1

    tokenizer = LanguageTokenizer()
    eval_ds = val_ds if len(val_ds) > 0 else train_ds
    cer = evaluate_cer(model, eval_ds, tokenizer, DEVICE)

    os.makedirs(CKPT_DIR, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "experiment": experiment,
            "accuracy": accuracy,
            "cer": cer,
        },
        f"{CKPT_DIR}/{experiment}_acc{accuracy}.pth",
    )
    return cer


# --- plots -------------------------------------------------------------------


def make_plots(results: dict):
    os.makedirs(FIG_DIR, exist_ok=True)

    # CER vs accuracy, one line per experiment
    plt.figure(figsize=(8, 5))
    for experiment in EXPERIMENTS:
        ys = [results[experiment][str(a)] for a in ACCURACIES]
        plt.plot(ACCURACIES, ys, marker="o", label=experiment)
    plt.axhline(0.30, color="gray", ls="--", lw=1, label="30% CER")
    plt.axhline(0.40, color="lightgray", ls="--", lw=1, label="40% CER")
    plt.xlabel("input-sequence accuracy")
    plt.ylabel("character error rate (CER)")
    plt.title("CER vs input accuracy, per experiment")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/cer_vs_accuracy.png", dpi=150)
    plt.close()

    # CER vs experiment at clean accuracy = 1.0
    plt.figure(figsize=(7, 5))
    xs = EXPERIMENTS
    ys = [results[e]["1.0"] for e in EXPERIMENTS]
    plt.bar(xs, ys, color=["#4c72b0", "#dd8452", "#55a868"])
    plt.ylabel("CER at accuracy = 1.0")
    plt.title("CER vs experiment (clean input)")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/cer_vs_experiment.png", dpi=150)
    plt.close()


def main():
    import sys

    # run a subset of experiments per invocation (fits the 10-min shell ceiling);
    # results accumulate in results.json so plots can be built once all are done
    experiments = sys.argv[1:] or EXPERIMENTS
    os.makedirs(FIG_DIR, exist_ok=True)
    results_path = f"{FIG_DIR}/results.json"

    results = {}
    if os.path.exists(results_path):
        with open(results_path) as f:
            results = json.load(f)

    for experiment in experiments:
        results.setdefault(experiment, {})
        for accuracy in ACCURACIES:
            if str(accuracy) in results[experiment]:  # resume: skip done runs
                print(f"[{experiment} acc={accuracy}] cached, skipping", flush=True)
                continue
            cer = train_one(experiment, accuracy)
            results[experiment][str(accuracy)] = cer
            print(f"[{experiment} acc={accuracy}] CER={cer:.3f}", flush=True)
            with open(results_path, "w") as f:  # checkpoint after each run
                json.dump(results, f, indent=2)

    # build plots once every experiment has a full accuracy sweep
    if all(e in results and len(results[e]) == len(ACCURACIES) for e in EXPERIMENTS):
        make_plots(results)
        print(
            f"\nAll experiments done. Saved plots + results to {FIG_DIR}/", flush=True
        )
    else:
        done = {e: len(results.get(e, {})) for e in EXPERIMENTS}
        print(
            f"\nProgress so far: {done}. Run remaining experiments to finish plots.",
            flush=True,
        )

    upload_artifacts()


def upload_artifacts():
    """Upload checkpoints + results + plots to Azure Blob (used by the A100 job)."""
    conn = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
    container = os.environ.get("BLOB_CONTAINER")
    if not (conn and container):
        return
    from azure.storage.blob import BlobServiceClient

    client = BlobServiceClient.from_connection_string(conn)
    try:
        client.create_container(container)
    except Exception:
        pass

    paths = []
    if os.path.isdir(CKPT_DIR):
        paths += [
            os.path.join(CKPT_DIR, f)
            for f in os.listdir(CKPT_DIR)
            if f.endswith(".pth")
        ]
    for f in ("results.json", "cer_vs_accuracy.png", "cer_vs_experiment.png"):
        p = os.path.join(FIG_DIR, f)
        if os.path.exists(p):
            paths.append(p)

    for path in paths:
        blob_name = f"lm_run/{os.path.basename(path)}"
        with open(path, "rb") as data:
            client.get_blob_client(container, blob_name).upload_blob(
                data, overwrite=True
            )
        print(f"uploaded {blob_name}", flush=True)


if __name__ == "__main__":
    main()
