"""
This script uses Beam Search and Trie-based decoding for evaluating language
model inference performance.
"""

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from eeg.language_model import LanguageDataset, LanguageModel, LanguageTokenizer
from eeg.trie import Trie

with open("config/language_model.yaml", "r") as config_file:
    config = yaml.safe_load(config_file)

    vocab_size = config["vocab_size"]
    num_layers = config["num_layers"]
    decoder_num_layers = config["decoder_num_layers"]
    num_heads = config["num_heads"]
    num_classes = config["num_classes"]
    embedding_dim = config["embedding_dim"]
    decoder_embedding_dim = config["decoder_embedding_dim"]
    ffn_hidden_dim = config["ffn_hidden_dim"]
    qk_length = config["qk_length"]
    value_length = config["value_length"]
    max_length = config["max_length"]

    encoder_dropout = config["encoder_dropout"]
    decoder_dropout = config["decoder_dropout"]

    recon_lambda = config["recon_lambda"]

    device = config["device"]
    batch_size = config["batch_size"]
    warmup_steps = config["warmup_steps"]
    base_lr = float(config["base_lr"])
    epochs = config["epochs"]

    run_name = config["run_name"]
    use_ckpt_path = config["use_ckpt_path"]
    save_ckpt_path = config["save_ckpt_path"]
    save_every = config["save_every"]

trie = Trie()

val_language_dataset = LanguageDataset(
    features_sequences=torch.tensor(0),
    mode="val",
    print_shapes=True,
)

val_language_dataloader = DataLoader(val_language_dataset, batch_size=32, shuffle=False)

model = LanguageModel(
    vocab_size=vocab_size,
    num_layers=num_layers,
    decoder_num_layers=decoder_num_layers,
    num_heads=num_heads,
    embedding_dim=embedding_dim,
    decoder_embedding_dim=decoder_embedding_dim,
    ffn_hidden_dim=ffn_hidden_dim,
    encoder_dropout=encoder_dropout,
    decoder_dropout=decoder_dropout,
).to(device)

tokenizer = LanguageTokenizer()

if use_ckpt_path is not None:
    state_dict = torch.load(use_ckpt_path, map_location=device)
    model.load_state_dict(state_dict["model"])


def validate(beam_width: int):
    """
    Run passes over the validation set and use beam search and tri-based
    decoding.
    """

    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for i, (feature, feature_mask, label, label_mask) in enumerate(
            tqdm(val_language_dataloader)
        ):
            feature: torch.Tensor = feature.to(device)
            feature_mask: torch.Tensor = feature_mask.to(device).bool()
            label: torch.Tensor = label.to(device).to(torch.int64)
            label_mask: torch.Tensor = label_mask.to(device).bool()

            batch_size = feature.size(0)  # get current batch size
            seq_len = feature.size(1)  # get max sequence length

            in_feature = feature[:, :-1, :]
            in_feature_mask = feature_mask[:, :-1]
            in_label = label[:, :1]  # start with <SOS>, shape (B, 1)
            in_label_mask = label_mask[:, :1]  # start with <SOS>

            # probability distribution for the next token
            label_logits, recon = model(
                src=in_feature,
                tgt=in_label,
                src_pad_mask=~in_feature_mask,  # flip because 1 should mean padding
                tgt_pad_mask=~in_label_mask,
            )  # out: (B, 1, vocab_size)

            # get logits for the first predicted token after <SOS>
            # remove the sequence length dimension because it is only 1
            next_token_logits = label_logits[:, -1, :]  # (B, vocab_size)

            # find the top K possible first tokens for each sequence in the batch
            # these become our initial beams
            top_scores, top_indices = torch.topk(
                next_token_logits,
                k=beam_width,
                dim=-1,
            )  # top_scores: (B, K), top_indices: (B, K)

            # initialize beam sequences
            # each beam starts with the first predicted token
            # shape: (B, K, 1)
            path_backpointers = top_indices.unsqueeze(-1)

            # initialize beam scores
            # each beam has one accumulated log probability score
            # shape: (B, K)
            sequence_scores = torch.log_softmax(next_token_logits, dim=-1).gather(
                dim=-1,
                index=top_indices,
            )
            # track which beams have already generated <EOS>
            # False means this beam is still generating
            finished = path_backpointers.squeeze(-1).eq(2)  # (B, K)

            # tile features
            # must repeat source features K times so they match our expanded beams
            # (B, seq_len, dim) -> (B * K, seq_len, dim)
            expanded_feature = feature[:, :-1, :].repeat_interleave(beam_width, dim=0)
            expanded_feature_mask = feature_mask[:, :-1].repeat_interleave(
                beam_width, dim=0
            )

            # beam search loop
            for step in range(1, seq_len):
                # flatten the input sequences from (B, K, step) to (B * K, step)
                decoder_input = path_backpointers.view(batch_size * beam_width, -1)

                # add <SOS> token back to front of generated paths
                sos_tokens = label[:, :1].repeat_interleave(beam_width, dim=0)
                decoder_input = torch.cat([sos_tokens, decoder_input], dim=1)

                # dummy padding mask for the target (all True since real tokens)
                tgt_mask = torch.ones_like(decoder_input).bool()

                # run model on all B*K paths simultaneously
                logits, recon = model(
                    src=expanded_feature,
                    tgt=decoder_input,
                    src_pad_mask=~expanded_feature_mask,
                    tgt_pad_mask=~tgt_mask,
                )  # (B * K, current_seq_len, vocab_size)

                # get logits for last token
                next_logits = logits[:, -1, :]  # (B * K, vocab_size)

                # reshape back to batches: (B, K, vocab_size)
                next_logits = next_logits.view(batch_size, beam_width, -1)

                log_probs = torch.log_softmax(next_logits, dim=-1)
                finished_mask = finished.unsqueeze(-1)
                frozen_log_probs = torch.full_like(log_probs, float("-inf"))
                frozen_log_probs[:, :, 2] = 0.0
                log_probs = torch.where(finished_mask, frozen_log_probs, log_probs)

                # add log probabilities
                new_scores = (
                    sequence_scores.unsqueeze(-1) + log_probs
                )  # (B, K, vocab_size)

                new_scores = new_scores.view(batch_size, -1)  # (B, K * vocab_size)

                # pick the top K survivors out of all possibilities
                top_scores, top_flat_indices = torch.topk(
                    new_scores, k=beam_width
                )  # (B, K)

                # convert flattened index back into:
                # old beam index and new token index
                old_beam_indices = top_flat_indices // vocab_size
                next_token_indices = top_flat_indices % vocab_size

                # determine which new beams have reached <EOS>
                new_finished = next_token_indices.eq(2)  # (B, K)

                # gather the finished status of the old beams that survived
                finished = torch.gather(
                    finished,
                    dim=1,
                    index=old_beam_indices,
                )

                # update finished status
                # once a beam reaches EOS, it stays finished
                finished = finished | new_finished

                # gather previous sequences that survived
                path_backpointers = torch.gather(
                    path_backpointers,
                    dim=1,
                    index=old_beam_indices.unsqueeze(-1).expand(
                        -1, -1, path_backpointers.size(-1)
                    ),
                )

                # append the newly predicted token
                path_backpointers = torch.cat(
                    [
                        path_backpointers,
                        next_token_indices.unsqueeze(-1),
                    ],
                    dim=-1,
                )

                # update scores for next iteration
                sequence_scores = top_scores

                if finished.all():
                    break

            # path_backpointers: (B, K, generated_length)
            # sequence_scores: (B, K)

            # choose the highest scoring beam for each sample
            best_beam = sequence_scores.argmax(dim=1)  # (B)

            # gather the best sequences
            predictions = torch.gather(
                path_backpointers,
                dim=1,
                index=best_beam[:, None, None].expand(
                    -1,
                    1,
                    path_backpointers.size(-1),
                ),
            ).squeeze(1)  # (B, generated_length)

            all_predictions.extend(predictions)
            all_labels.extend(label)

    for pred_sequence, label_sequence in zip(all_predictions, all_labels):
        decoded_pred = tokenizer.decode(pred_sequence)
        decoded_labels = tokenizer.decode(label_sequence)

        print(decoded_pred)
        print(f"{decoded_labels}\n")


validate(beam_width=5)
