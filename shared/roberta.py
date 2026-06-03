from transformers import RobertaForMaskedLM, RobertaTokenizer


def load_roberta(model_name: str, save_dir: str):
    """
    Downloads Roberta from HuggingFace hub (if not cached), saves locally,
    reloads from disk, and prints parameter count.

    RobertaTokenizer (slow) is intentional. WWM needs the Ġ prefix.
    """
    print(f"\n── Downloading model and tokenizer: {model_name} ──")

    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaForMaskedLM.from_pretrained(model_name)

    tokenizer.save_pretrained(save_dir)
    model.save_pretrained(save_dir)

    # Reload from local path to confirm save is clean
    tokenizer = RobertaTokenizer.from_pretrained(save_dir)
    model = RobertaForMaskedLM.from_pretrained(save_dir)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded from local | Params: {total_params:,}")

    return model, tokenizer
