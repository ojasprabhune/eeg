"""
This script uses Beam Search and Trie-based decoding for evaluating language
model inference performance.
"""

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from eeg.language_model import (
    LanguageDataset,
    LanguageModel,
    LanguageTokenizer,
    compute_cer,
    compute_wer,
    words_from_lengths,
)
from eeg.trie import Trie, TrieNode

# =============================================================================
# CONFIGURATION
# =============================================================================

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

    min_value = config["min_value"]
    k = config["k"]

    run_name = config["run_name"]
    use_ckpt_path = config["use_ckpt_path"]
    save_ckpt_path = config["save_ckpt_path"]
    save_every = config["save_every"]

    beam_width = config["beam_width"]

# =============================================================================
# DATA
# =============================================================================

trie = Trie()

val_language_dataset = LanguageDataset(
    num_classes=num_classes,
    mode="val",
    print_shapes=True,
)

val_language_dataloader = DataLoader(val_language_dataset, batch_size=32, shuffle=False)

# =============================================================================
# MODEL
# =============================================================================

model = LanguageModel(
    vocab_size=vocab_size,
    num_layers=num_layers,
    decoder_num_layers=decoder_num_layers,
    num_heads=num_heads,
    num_inputs_classes=num_classes,
    embedding_dim=embedding_dim,
    decoder_embedding_dim=decoder_embedding_dim,
    ffn_hidden_dim=ffn_hidden_dim,
    encoder_dropout=encoder_dropout,
    decoder_dropout=decoder_dropout,
    min_value=min_value,
    k=k,
).to(device)

tokenizer = LanguageTokenizer()

if use_ckpt_path is not None:
    state_dict = torch.load(use_ckpt_path, map_location=device)
    model.load_state_dict(state_dict["model"])
else:
    raise ValueError("no checkpoint given to load")

# =============================================================================
# REGULAR VALIDATION
# =============================================================================


def validate(
    sample_strategy: str, temperature: float, print_sequences: bool, print_words: bool
):
    """
    Validates language model using autoregressive decoding and regalar sampling
    strategies.

    sample_strategy:
        "greedy"        -> argmax sampling
        "multinomial"   -> multinomial sampling
    """

    model.eval()

    all_predictions = []
    all_labels = []
    all_word_lengths = []

    step = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for feature, feature_mask, label, label_mask, word_lengths in tqdm(
            val_language_dataloader
        ):
            feature: torch.Tensor = feature.to(device)
            feature_mask: torch.Tensor = feature_mask.to(device).bool()
            label: torch.Tensor = label.to(device).to(torch.int64)
            label_mask: torch.Tensor = label_mask.to(device).bool()

            batch_size = feature.size(0)
            seq_len = label.size(1)

            in_feature = feature[:, :-1, :]
            in_feature_mask = feature_mask[:, :-1]
            predictions = label[:, :1]  # <SOS>
            prediction_mask = torch.ones_like(predictions).bool()  # all tokens

            finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

            for _ in range(1, seq_len):
                label_logits, _ = model(
                    src=in_feature,
                    tgt=predictions,
                    src_pad_mask=~in_feature_mask,
                    tgt_pad_mask=~prediction_mask,
                    step=step,
                    return_epsilon=False,
                    use_scheduled_sampling=False,
                )

                # get logits for the next predicted token
                next_token_logits = label_logits[:, -1, :]  # (B, vocab_size)

                if sample_strategy == "greedy":
                    # select the highest probability token
                    next_token = next_token_logits.argmax(dim=-1)

                elif sample_strategy == "multinomial":
                    # sample according to the predicted probability distribution
                    probs = torch.softmax(next_token_logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)

                else:
                    raise ValueError(
                        "sample_strategy must be 'greedy' or 'multinomial'"
                    )

                # once a sequence reaches <EOS>, keep emitting <EOS>. both
                # next_token and finished are (B,)
                next_token = torch.where(
                    finished,
                    torch.full_like(next_token, 2),  # <EOS>
                    next_token,
                )

                # append the predicted token
                predictions = torch.cat(
                    [predictions, next_token.unsqueeze(1)],
                    dim=1,
                )

                # update the target padding mask
                prediction_mask = torch.ones_like(predictions).bool()

                # update finished status
                finished |= next_token.eq(2)

                step += 1

            # remove the initial <SOS> so predictions align with gt labels
            predictions = predictions[:, 1:]
            valid = label_mask[:, 1:]

            correct += ((predictions == label[:, 1:]) & valid).sum().item()
            total += valid.sum().item()

            all_predictions.extend(predictions)  # remove <SOS>
            all_labels.extend(label)
            all_word_lengths.extend(word_lengths)

    total_cer = 0.0
    total_wer = 0.0
    num_sentences = 0

    for pred_sequence, label_sequence, sentence_word_lengths in zip(
        all_predictions, all_labels, all_word_lengths
    ):
        decoded_pred = tokenizer.decode(pred_sequence)
        decoded_labels = tokenizer.decode(label_sequence)

        pred_sentence = decoded_pred[0]
        true_sentence = decoded_labels[0]

        lengths = [
            int(length.item()) for length in sentence_word_lengths if length.item() > 0
        ]

        true_words = words_from_lengths(true_sentence, lengths)
        pred_words = words_from_lengths(pred_sentence, lengths)

        if print_sequences:
            if print_words:
                print(pred_words)
                print(true_words)
                print()
            else:
                print(pred_sentence)
                print(true_sentence)
                print()

        total_cer += compute_cer(pred_sentence, true_sentence)
        total_wer += compute_wer(pred_words, true_words)
        num_sentences += 1

    accuracy = correct / max(total, 1)

    print(f"\n---- regular validation ({sample_strategy}) ----")
    print(f"Raw accuracy: {accuracy:.2%}")
    print(f"Average CER: {total_cer / max(num_sentences, 1):.2f}")
    print(f"Average WER: {total_wer / max(num_sentences, 1):.2f}\n")


# =============================================================================
# BEAM SEARCH VALIDATION
# =============================================================================


def beam_search_validate(beam_width: int, print_sequences: bool, print_words: bool):
    """
    Validates language model using beam search inference.
    """

    model.eval()

    all_predictions = []
    all_labels = []
    all_word_lengths = []

    step = 0

    with torch.no_grad():
        for feature, feature_mask, label, label_mask, word_lengths in tqdm(
            val_language_dataloader
        ):
            feature: torch.Tensor = feature.to(device)
            feature_mask: torch.Tensor = feature_mask.to(device).bool()
            label: torch.Tensor = label.to(device).to(torch.int64)
            label_mask: torch.Tensor = label_mask.to(device).bool()

            batch_size = feature.size(0)  # get current batch size
            seq_len = label.size(1)  # get max sequence length

            in_feature = feature[:, :-1, :]
            in_feature_mask = feature_mask[:, :-1]
            in_label = label[:, :1]  # start with <SOS>, shape (B, 1)
            in_label_mask = label_mask[:, :1]  # start with <SOS>

            # probability distribution for the next token
            label_logits, _ = model(
                src=in_feature,
                tgt=in_label,
                src_pad_mask=~in_feature_mask,  # flip because 1 should mean padding
                tgt_pad_mask=~in_label_mask,
                step=step,
                return_epsilon=False,
                use_scheduled_sampling=False,
            )  # out: (B, 1, vocab_size)

            step += 1

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
            for _ in range(1, seq_len):
                # flatten the input sequences from (B, K, step) to (B * K, step)
                decoder_input = path_backpointers.view(batch_size * beam_width, -1)

                # add <SOS> token back to front of generated paths
                sos_tokens = label[:, :1].repeat_interleave(beam_width, dim=0)
                decoder_input = torch.cat([sos_tokens, decoder_input], dim=1)

                # dummy padding mask for the target (all True since real tokens)
                tgt_mask = torch.ones_like(decoder_input).bool()

                # run model on all B*K paths simultaneously
                logits, _ = model(
                    src=expanded_feature,
                    tgt=decoder_input,
                    src_pad_mask=~expanded_feature_mask,
                    tgt_pad_mask=~tgt_mask,
                    step=step,
                    return_epsilon=False,
                    use_scheduled_sampling=False,
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

                step += 1

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
            all_word_lengths.extend(word_lengths)

    total_cer = 0.0
    total_wer = 0.0
    num_sentences = 0

    for pred_sequence, label_sequence, sentence_word_lengths in zip(
        all_predictions, all_labels, all_word_lengths
    ):
        decoded_pred = tokenizer.decode(pred_sequence)
        decoded_labels = tokenizer.decode(label_sequence)

        pred_sentence = decoded_pred[0]
        true_sentence = decoded_labels[0]

        # strip the zero-padding off the word length row to get the real
        # word boundaries for this sentence
        lengths = [
            int(length.item()) for length in sentence_word_lengths if length.item() > 0
        ]

        true_words = words_from_lengths(true_sentence, lengths)
        pred_words = words_from_lengths(pred_sentence, lengths)

        if print_sequences:
            if print_words:
                print(pred_words)
                print(true_words)
                print()
            else:
                print(pred_sentence)
                print(true_sentence)
                print()

        total_cer += compute_cer(pred_sentence, true_sentence)
        total_wer += compute_wer(pred_words, true_words)
        num_sentences += 1

    print(f"\n---- beam search validation ({beam_width} width) ----")
    print(f"Average CER: {total_cer / max(num_sentences, 1):.2f}")
    print(f"Average WER: {total_wer / max(num_sentences, 1):.2f}\n")


# =============================================================================
# BEAM SEARCH & TRIE-BASED DECODING VALIDATION
# =============================================================================


def _allowed_token_ids(
    node: TrieNode,
    trie: Trie,
    tokenizer: LanguageTokenizer,
    allow_eos: bool,
) -> list[int]:
    """
    Given the trie node a beam currently sits at (its position within the
    word it's mid-typing), returns the token ids it's allowed to produce
    next: letters that continue the current word, and, if the current node
    is already a complete word, letters that start a brand new word plus
    (optionally) <EOS> to end the sentence there.
    """
    letters = set(node.children.keys())  # continue the current word

    if node.is_word:
        letters |= set(trie.root.children.keys())  # or start a new word

    allowed_ids = [tokenizer.letter_to_token[letter] for letter in letters]

    if node.is_word and allow_eos:
        allowed_ids.append(tokenizer.letter_to_token["<EOS>"])

    if not allowed_ids:
        # true dead end (shouldn't happen if every prior step was masked
        # correctly, but guard against NaNs from an all -inf row anyway):
        # force this beam to terminate instead of corrupting the softmax
        allowed_ids = [tokenizer.letter_to_token["<EOS>"]]

    return allowed_ids


def _step_trie_node(node: TrieNode, trie: Trie, letter: str) -> TrieNode:
    """
    Advances a single beam's trie-node state by one chosen letter: continues
    into the current word if possible, otherwise treats the letter as the
    start of a new word (only reachable if the current node was a completed
    word, which is exactly what _allowed_token_ids permits).
    """
    if letter in node.children:
        return node.children[letter]  # continuing the current word
    if node.is_word and letter in trie.root.children:
        return trie.root.children[letter]  # starting a new word
    return trie.root.children.get(letter, trie.root)  # guard, shouldn't hit


def beam_trie_validate(beam_width: int, print_sequences: bool, print_words: bool):
    """
    Validates language model using beam search inferenced with trie-based decoding.
    """

    model.eval()

    all_predictions = []
    all_labels = []
    all_word_lengths = []

    step = 0

    with torch.no_grad():
        for feature, feature_mask, label, label_mask, word_lengths in tqdm(
            val_language_dataloader
        ):
            feature: torch.Tensor = feature.to(device)
            feature_mask: torch.Tensor = feature_mask.to(device).bool()
            label: torch.Tensor = label.to(device).to(torch.int64)
            label_mask: torch.Tensor = label_mask.to(device).bool()

            batch_size = feature.size(0)  # get current batch size
            seq_len = label.size(1)  # get max sequence length

            in_feature = feature[:, :-1, :]
            in_feature_mask = feature_mask[:, :-1]
            in_label = label[:, :1]  # start with <SOS>, shape (B, 1)
            in_label_mask = label_mask[:, :1]  # start with <SOS>

            # probability distribution for the next token
            label_logits, _ = model(
                src=in_feature,
                tgt=in_label,
                src_pad_mask=~in_feature_mask,  # flip because 1 should mean padding
                tgt_pad_mask=~in_label_mask,
                step=step,
                return_epsilon=False,
                use_scheduled_sampling=False,
            )  # out: (B, 1, vocab_size)

            step += 1

            next_token_logits = label_logits[:, -1, :]  # (B, vocab_size)

            # trie mask: every beam sits at the trie root at the first
            # step (no letters generated yet), so mask to whatever letters
            # can legally start a word. <EOS> can't be legal yet since no
            # beam has completed a word.
            trie = val_language_dataset.trie
            start_ids = _allowed_token_ids(trie.root, trie, tokenizer, allow_eos=False)
            trie_mask = torch.zeros_like(next_token_logits, dtype=torch.bool)
            trie_mask[:, start_ids] = True  # same start-set for every sample in batch

            next_token_logits = next_token_logits.masked_fill(~trie_mask, float("-inf"))

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

            # per-beam trie state: track which trie node each beam
            # currently sits at (its position within the word it's mid-
            # typing). every beam starts at the trie root, then advances by
            # the letter it just picked.
            beam_nodes: list[list[TrieNode]] = [
                [
                    _step_trie_node(
                        trie.root,
                        trie,
                        tokenizer.token_to_letter[int(top_indices[b, k].item())],
                    )
                    for k in range(beam_width)
                ]
                for b in range(batch_size)
            ]

            # tile features
            # must repeat source features K times so they match our expanded beams
            # (B, seq_len, dim) -> (B * K, seq_len, dim)
            expanded_feature = feature[:, :-1, :].repeat_interleave(beam_width, dim=0)
            expanded_feature_mask = feature_mask[:, :-1].repeat_interleave(
                beam_width, dim=0
            )

            # beam search loop
            for _ in range(1, seq_len):
                # flatten the input sequences from (B, K, step) to (B * K, step)
                decoder_input = path_backpointers.view(batch_size * beam_width, -1)

                # add <SOS> token back to front of generated paths
                sos_tokens = label[:, :1].repeat_interleave(beam_width, dim=0)
                decoder_input = torch.cat([sos_tokens, decoder_input], dim=1)

                # dummy padding mask for the target (all True since real tokens)
                tgt_mask = torch.ones_like(decoder_input).bool()

                # run model on all B*K paths simultaneously
                logits, _ = model(
                    src=expanded_feature,
                    tgt=decoder_input,
                    src_pad_mask=~expanded_feature_mask,
                    tgt_pad_mask=~tgt_mask,
                    step=step,
                    return_epsilon=False,
                    use_scheduled_sampling=False,
                )  # (B * K, current_seq_len, vocab_size)

                # get logits for last token
                next_logits = logits[:, -1, :]  # (B * K, vocab_size)

                # reshape back to batches: (B, K, vocab_size)
                next_logits = next_logits.view(batch_size, beam_width, -1)

                log_probs = torch.log_softmax(next_logits, dim=-1)  # (B, K, vocab_size)
                finished_mask = finished.unsqueeze(-1)
                frozen_log_probs = torch.full_like(log_probs, float("-inf"))
                frozen_log_probs[:, :, 2] = 0.0
                log_probs = torch.where(finished_mask, frozen_log_probs, log_probs)

                # trie mask: for every (batch, beam) pair, use its
                # current trie-node state to figure out which letters
                # legally continue the word it's mid-typing, or start a new
                # one / end the sentence if it just completed a word.
                # finished beams are exempted so they can keep emitting
                # <EOS> via frozen_log_probs above.
                trie_mask = torch.zeros_like(log_probs, dtype=torch.bool)
                for b in range(batch_size):
                    for k in range(beam_width):
                        allowed_ids = _allowed_token_ids(
                            beam_nodes[b][k], trie, tokenizer, allow_eos=True
                        )
                        trie_mask[b, k, allowed_ids] = True

                trie_mask = trie_mask | finished.unsqueeze(-1)
                log_probs = log_probs.masked_fill(~trie_mask, float("-inf"))

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

                # --- advance per-beam trie state: reorder to match the
                # surviving beams (same reordering path_backpointers just
                # went through), then step each by the letter it just
                # picked. finished beams don't need a real update since
                # they're masked out from here on regardless.
                old_beam_nodes = beam_nodes
                beam_nodes = []
                for b in range(batch_size):
                    row = []
                    for k in range(beam_width):
                        old_k = old_beam_indices[b, k].item()
                        node = old_beam_nodes[b][old_k]
                        letter = tokenizer.token_to_letter[
                            next_token_indices[b, k].item()
                        ]
                        if letter not in ("<EOS>", "<PAD>", "<SOS>"):
                            node = _step_trie_node(node, trie, letter)
                        row.append(node)
                    beam_nodes.append(row)

                # update scores for next iteration
                sequence_scores = top_scores

                if finished.all():
                    break

                step += 1

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
            all_word_lengths.extend(word_lengths)

    total_cer = 0.0
    total_wer = 0.0
    num_sentences = 0

    for pred_sequence, label_sequence, sentence_word_lengths in zip(
        all_predictions, all_labels, all_word_lengths
    ):
        decoded_pred = tokenizer.decode(pred_sequence)
        decoded_labels = tokenizer.decode(label_sequence)

        pred_sentence = decoded_pred[0]
        true_sentence = decoded_labels[0]

        # strip the zero-padding off the word length row to get the real
        # word boundaries for this sentence
        lengths = [
            int(length.item()) for length in sentence_word_lengths if length.item() > 0
        ]

        true_words = words_from_lengths(true_sentence, lengths)
        pred_words = words_from_lengths(pred_sentence, lengths)

        if print_sequences:
            if print_words:
                print(pred_words)
                print(true_words)
                print()
            else:
                print(pred_sentence)
                print(true_sentence)
                print()

        total_cer += compute_cer(pred_sentence, true_sentence)
        total_wer += compute_wer(pred_words, true_words)
        num_sentences += 1

    print(f"\n---- beam trie search validation ({beam_width} width) ----")
    print(f"Average CER: {total_cer / max(num_sentences, 1):.2f}")
    print(f"Average WER: {total_wer / max(num_sentences, 1):.2f}\n")


validate(
    sample_strategy="greedy",
    temperature=1,
    print_sequences=True,
    print_words=True,
)
beam_search_validate(
    beam_width=beam_width,
    print_sequences=False,
    print_words=False,
)
beam_trie_validate(
    beam_width=beam_width,
    print_sequences=False,
    print_words=False,
)
