"""
Compare teacher-forced logits (ground truth tgt) against autoregressive
logits (model's own generated prefix) on one validation sample, to see
exactly where the two diverge.
"""

import torch
import yaml

from eeg.language_model import LanguageDataset, LanguageModel, LanguageTokenizer

with open("config/language_model.yaml", "r") as config_file:
    config = yaml.safe_load(config_file)

val_language_dataset = LanguageDataset(
    num_classes=config["num_classes"], mode="val", print_shapes=False
)
tokenizer = LanguageTokenizer()

model = LanguageModel(
    vocab_size=config["vocab_size"],
    num_layers=config["num_layers"],
    decoder_num_layers=config["decoder_num_layers"],
    num_heads=config["num_heads"],
    num_inputs_classes=config["num_classes"],
    embedding_dim=config["embedding_dim"],
    decoder_embedding_dim=config["decoder_embedding_dim"],
    ffn_hidden_dim=config["ffn_hidden_dim"],
    encoder_dropout=config["encoder_dropout"],
    decoder_dropout=config["decoder_dropout"],
    min_value=config["min_value"],
    k=config["k"],
).to(config["device"])

state_dict = torch.load(config["use_ckpt_path"], map_location=config["device"])
model.load_state_dict(state_dict["model"])
model.eval()

# --- grab one sample -------------------------------------------------------
feature, feature_mask, label, label_mask, _ = val_language_dataset[0]
feature = feature.unsqueeze(0).to(config["device"])
feature_mask = feature_mask.unsqueeze(0).to(config["device"]).bool()
label = label.unsqueeze(0).to(config["device"]).to(torch.int64)
label_mask = label_mask.unsqueeze(0).to(config["device"]).bool()

in_feature = feature[:, :-1, :]
in_feature_mask = feature_mask[:, :-1]

# --- teacher-forced pass: feed the real label, one shot ---------------------
in_label = label[:, :-1]
in_label_mask = label_mask[:, :-1]

with torch.no_grad():
    tf_logits, _ = model(
        src=in_feature,
        tgt=in_label,
        src_pad_mask=~in_feature_mask,
        tgt_pad_mask=~in_label_mask,
        step=0,
        return_epsilon=False,
        use_scheduled_sampling=False,
    )

tf_predictions = tf_logits.argmax(dim=-1)  # (1, T)

# --- autoregressive pass: same loop as regular_validate's greedy path -------
predictions = label[:, :1]  # <SOS>
prediction_mask = torch.ones_like(predictions).bool()
seq_len = label.size(1)

with torch.no_grad():
    for _ in range(seq_len):
        ar_logits, _ = model(
            src=in_feature,
            tgt=predictions,
            src_pad_mask=~in_feature_mask,
            tgt_pad_mask=~prediction_mask,
            step=0,
            return_epsilon=False,
            use_scheduled_sampling=False,
        )
        next_token = ar_logits[:, -1, :].argmax(dim=-1)
        predictions = torch.cat([predictions, next_token.unsqueeze(1)], dim=1)
        prediction_mask = torch.ones_like(predictions).bool()

ar_predictions = predictions[:, 1:]  # drop <SOS>

print("ground truth:   ", tokenizer.decode(label[0])[0])
print("teacher-forced: ", tokenizer.decode(tf_predictions[0])[0])
print("autoregressive: ", tokenizer.decode(ar_predictions[0])[0])
print()
print("tf argmax:", tf_predictions[0].tolist())
print("ar argmax:", ar_predictions[0].tolist())
