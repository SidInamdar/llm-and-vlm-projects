import math
import random

import numpy as np
import torch


def compute_mlm_perplexity(eval_pred):
    """
    Derives perplexity from the cross-entropy loss over masked positions.
    Trainer has already computed eval_loss before calling this function.

    Args:
        eval_pred : EvalPrediction(predictions=logits, label_ids=labels)

    Returns:
        dict with perplexity
    """
    logits, labels = eval_pred

    # Recompute loss here from logits and labels
    # ignore_index=-100 means unmasked positions don't contribute
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
    loss = loss_fct(
        torch.tensor(logits).view(-1, logits.shape[-1]),
        torch.tensor(labels).view(-1),
    )
    perplexity = math.exp(loss.item())
    return {"perplexity": round(perplexity, 4)}


def token_length_distribution(sentences, tokenizer, sample_size, max_length, writer=None):
    """
    Tokenise a sample of sentences without padding to see actual lengths.
    Confirms max_length covers the corpus adequately.
    If p95 > max_length, too many sentences are being truncated — raise max_length.
    """
    print("\n── Checking token length distribution ──")

    sample_for_dist = random.sample(sentences, min(sample_size, len(sentences)))

    lengths = [
        len(tokenizer(s, truncation=False, add_special_tokens=True)["input_ids"])
        for s in sample_for_dist
    ]

    p50 = int(np.percentile(lengths, 50))
    p90 = int(np.percentile(lengths, 90))
    p95 = int(np.percentile(lengths, 95))
    p99 = int(np.percentile(lengths, 99))
    pmax = max(lengths)
    pct_covered = sum(length <= max_length for length in lengths) / len(lengths) * 100

    print(f"  p50={p50} | p90={p90} | p95={p95} | p99={p99} | max={pmax}")
    print(f"  max_length={max_length} covers {pct_covered:.1f}% without truncation")

    if pct_covered < 90:
        print(f"  WARNING: only {pct_covered:.1f}% covered — consider raising max_length")

    if writer:
        writer.add_scalar("data/token_len_p50", p50, 0)
        writer.add_scalar("data/token_len_p90", p90, 0)
        writer.add_scalar("data/token_len_p95", p95, 0)
        writer.add_scalar("data/token_len_p99", p99, 0)
        writer.add_scalar("data/token_len_max", pmax, 0)
        writer.add_scalar("data/pct_no_truncation", round(pct_covered, 2), 0)

    return {"p50": p50, "p90": p90, "p95": p95, "p99": p99, "max": pmax, "pct_covered": pct_covered}
