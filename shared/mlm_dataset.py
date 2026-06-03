from datasets import Dataset


def tokenise_sentences(sentences, tokenizer, max_length):
    """
    Tokenise a list of sentences into a HuggingFace Dataset.
    Returns Dataset with input_ids, attention_mask, special_tokens_mask.
    Masking is NOT applied here — that happens in the collator.

    Args:
        sentences  : list of sentence strings
        tokenizer  : RobertaTokenizer instance
        max_length : maximum token length (pad and truncate to this)

    Returns:
        HuggingFace Dataset
    """
    encoding = tokenizer(
        sentences,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_special_tokens_mask=True,  # required by DataCollatorForWholeWordMask
    )
    return Dataset.from_dict(
        {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "special_tokens_mask": encoding["special_tokens_mask"],
        }
    )
