import re
from collections import defaultdict

import torch
from transformers import pipeline


def mine_verification_sentences(sentences, terms, tokenizer, per_term=2):
    """
    For each term in terms, find sentences from the corpus that:
      1. Contain the term as a whole word (not substring)
      2. Are long enough to provide meaningful context (>= 12 words)

    Then replace the term with <mask> to create the fill-mask input.
    Returns a list of dicts with sentence, masked_sentence, ground_truth, domain.

    Args:
        sentences  : list of sentence strings to search
        terms      : list of domain term strings to look for
        tokenizer  : RobertaTokenizer instance
        per_term   : how many sentences to mine per term

    Returns:
        List of verification item dicts
    """
    # Group candidate sentences by term
    candidates = defaultdict(list)

    for sent in sentences:
        sent_lower = sent.lower()
        for term in terms:
            # Whole word match — avoids "crankshaft" matching inside "crankshafts"
            pattern = r"\b" + re.escape(term.lower()) + r"\b"
            if re.search(pattern, sent_lower) and len(sent.split()) >= 12:
                candidates[term].append(sent)

    verification_items = []
    for term in terms:
        if not candidates[term]:
            print(f"  WARNING: no sentences found for term '{term}'")
            continue

        # Sort by length descending — longer context is a harder, more meaningful test
        sorted_cands = sorted(candidates[term], key=lambda s: len(s), reverse=True)

        # Take top per_term unique sentences
        selected = sorted_cands[:per_term]

        for sent in selected:
            # Replace the term (case-insensitive) with <mask>
            # re.sub with IGNORECASE handles "Injector", "INJECTOR" etc.
            masked = re.sub(
                r"\b" + re.escape(term) + r"\b",
                tokenizer.mask_token,  # <mask>
                sent,
                count=1,  # replace only the first occurrence
                flags=re.IGNORECASE,
            )
            verification_items.append(
                {
                    "original": sent,
                    "masked_sentence": masked,
                    "ground_truth": term,
                    "domain": term,  # use term as domain label
                }
            )
            print(f"  [{term}] {masked[:90]}...")

    print(f"\nTotal verification sentences mined: {len(verification_items)}")
    return verification_items


def run_benchmarks(
    model, tokenizer, verification_items, device, verification_terms, max_length, label="baseline"
):
    """
    Run all verification signals on the given model.
    Returns a dict of results serialisable to JSON.

    Args:
        model             : RobertaForMaskedLM instance
        tokenizer         : RobertaTokenizer instance
        verification_items: list of dicts from mine_verification_sentences
        device            : torch device
        verification_terms: list of domain terms
        max_length        : maximum token length
        label             : string tag for logging ("baseline" or "finetuned")

    Returns:
        results dict with fill_mask, token_stats, embeddings
    """
    model.eval()

    # ── HuggingFace fill-mask pipeline ──────────────────────────────────────
    # Pipeline handles tokenisation, forward pass, softmax, top-k extraction.
    fill_pipe = pipeline(
        "fill-mask",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1,
        top_k=10,  # retrieve top-10 so we can check rank within top-10
    )

    fill_mask_results = {}
    token_stats = {}

    for item in verification_items:
        masked_sent = item["masked_sentence"]
        ground_truth = item["ground_truth"]

        # ── Signal 1: fill-mask top-10 ────────────────────────────────────
        preds = fill_pipe(masked_sent)
        top10_tokens = [p["token_str"].strip() for p in preds]
        top10_scores = [round(p["score"], 5) for p in preds]

        # Check if ground truth appears in top-10
        gt_in_top10 = next(
            (
                i + 1
                for i, p in enumerate(preds)
                if p["token_str"].strip().lower() == ground_truth.lower()
            ),
            None,  # None means not in top-10
        )

        fill_mask_results[masked_sent] = {
            "ground_truth": ground_truth,
            "top10_tokens": top10_tokens,
            "top10_scores": top10_scores,
            "gt_rank_top10": gt_in_top10,
        }

        # ── Signal 2 & 3: rank and probability over full vocab ────────────
        # Manually run forward pass to get full 50k-dim logit vector
        inputs = tokenizer(
            masked_sent, return_tensors="pt", truncation=True, max_length=max_length
        ).to(device)
        mask_pos = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]

        with torch.no_grad():
            logits = model(**inputs).logits  # [1, seq_len, vocab_size]

        # Softmax over vocab dimension at the masked position
        probs = torch.softmax(logits[0, mask_pos[0]], dim=-1)  # [vocab_size]
        sorted_ids = probs.argsort(descending=True)  # indices by prob

        # Ground truth token ID — use encode to handle subword correctly
        gt_ids = tokenizer.encode(
            " " + ground_truth,  # leading space → Ġ prefix → correct ID
            add_special_tokens=False,
        )
        gt_id = gt_ids[0] if gt_ids else tokenizer.convert_tokens_to_ids(ground_truth)

        # Rank: position in sorted list (1-indexed)
        rank_tensor = (sorted_ids == gt_id).nonzero(as_tuple=True)
        gt_rank = rank_tensor[0].item() + 1 if len(rank_tensor[0]) > 0 else -1
        gt_prob = probs[gt_id].item()

        # ── Signal 4: total domain mass ───────────────────────────────────
        # Sum probability across all verification terms — measures how much
        # of the model's probability mass sits on domain vocabulary
        domain_mass = 0.0
        for term in verification_terms:
            term_ids = tokenizer.encode(" " + term, add_special_tokens=False)
            term_id = term_ids[0] if term_ids else tokenizer.convert_tokens_to_ids(term)
            domain_mass += probs[term_id].item()

        token_stats[ground_truth] = {
            "gt_rank": gt_rank,
            "gt_prob": round(gt_prob, 7),
            "domain_mass": round(domain_mass, 6),
            "masked_sentence": masked_sent,
        }

        print(
            f"  [{label}] {ground_truth:15s} | "
            f"rank={gt_rank:5d} | prob={gt_prob:.6f} | "
            f"domain_mass={domain_mass:.4f} | "
            f"top3={top10_tokens[:3]}"
        )

    # ── Signal 5: [CLS] embeddings ────────────────────────────────────────
    # Collect embeddings for every verification sentence.
    # After training we run UMAP on baseline vs finetuned embeddings side by side.
    # model.roberta bypasses the MLM head — gives raw encoder [CLS] output.
    embeddings = {}
    for item in verification_items:
        inputs = tokenizer(
            item["original"],  # use original (unmasked) sentence
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding="max_length",
        ).to(device)

        with torch.no_grad():
            enc_out = model.roberta(**inputs)

        cls_vec = enc_out.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
        embeddings[item["original"]] = {
            "domain": item["domain"],
            "embedding": cls_vec.tolist(),
        }

    return {
        "label": label,
        "fill_mask": fill_mask_results,
        "token_stats": token_stats,
        "embeddings": embeddings,
    }
