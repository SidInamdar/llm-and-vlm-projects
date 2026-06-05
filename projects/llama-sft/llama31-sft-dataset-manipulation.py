from datasets import Dataset, DatasetDict, load_from_disk

multiwoz = load_from_disk("/home/siddhesh/Documents/repos/llm-and-vlm-projects/datasets/raw/multiwoz")
bitext = load_from_disk('/home/siddhesh/Documents/repos/llm-and-vlm-projects/datasets/raw/bitext')

def multiwoz_to_instruction_response(dialogue_turns):
    """
    Convert MultiWOZ dialogue turns to instruction-response pairs.
    Each USER turn paired with the immediately following SYSTEM turn.
    """
    pairs = []
    
    for i, turn in enumerate(dialogue_turns):
        if turn["speaker"] == "USER":
            # check if next turn exists and is SYSTEM
            if i + 1 < len(dialogue_turns) and dialogue_turns[i + 1]["speaker"] == "SYSTEM":
                pairs.append({
                    "instruction": turn["utterance"],
                    "response": dialogue_turns[i + 1]["utterance"]
                })
    
    return pairs

def process_multiwoz(dataset):
    all_pairs = []
    for dialogue in dataset:
        pairs = multiwoz_to_instruction_response(dialogue["turns"])
        all_pairs.extend(pairs)
    return all_pairs

def process_bitext(dataset):
    all_pairs = []
    for dialogue in dataset:
        all_pairs.append({
            "instruction": dialogue["instruction"],
            "response": dialogue["response"]
        })
    return all_pairs

multiwoz_pairs = process_multiwoz(multiwoz["train"])
bitext_pairs = process_bitext(bitext["train"])
combined_dataset = Dataset.from_list(bitext_pairs + multiwoz_pairs)

print(combined_dataset) 



#import fasttext
import urllib.request
import numpy as np
from nltk.metrics import jaccard_distance
from transformers import AutoTokenizer
from datasketch import MinHash, MinHashLSH

token_limit = 580
tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-3.1-8b",
    cache_dir="/home/siddhesh/Documents/repos/llm-and-vlm-projects/models/tokenizers",
)

urllib.request.urlretrieve(
    "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin",
    "lid.176.bin",
)
#english_detect_model = fasttext.load_model("lid.176.bin")
from lingua import Language, LanguageDetectorBuilder
english_detect_algo = LanguageDetectorBuilder.from_languages(
    Language.ENGLISH,
    Language.CHINESE,
    Language.FRENCH,
).build()

def wc_filter_function(sample: dict) -> bool:
    instruction = sample.get("instruction", "")
    response = sample.get("response", "")
    instruction_str = instruction if instruction is not None else ""
    response_str = response if response is not None else ""

    query_wc = len(instruction_str.split())
    resp_wc = len(response_str.split())
    return query_wc > 4 and resp_wc > 5


def tokenizer_filter_function(sample: dict) -> bool:
    instruction = sample.get("instruction", "")
    response = sample.get("response", "")

    tokens = tokenizer(
        instruction + " " + response,
        truncation=False,
        add_special_tokens=True,
    )
    return len(tokens["input_ids"]) <= token_limit


def filter_english(sample: dict) -> bool:
    text = sample.get("instruction", "") + sample.get("response", "")
    clean_text = text.replace("\n", "")
    # Workaround: fasttext returns a tuple from zip(*predictions); np.asarray
    # avoids the NumPy 2.0 ValueError raised by np.array(..., copy=False).
    #labels, scores = english_detect_model.predict(clean_text, k=1)
    #scores = np.asarray(scores)
    #return labels[0] == "__label__en" and scores[0] > 0.8
    lang = english_detect_algo.detect_language_of(clean_text)
    return lang == Language.ENGLISH

def filter_same_inst_resp_pairs(sample: dict, threshold: float = 0.85) -> bool:
    instruction = sample.get("instruction", "").lower().split()
    response = sample.get("response", "").lower().split()
    if not instruction or not response:
        return False
    similarity = 1 - jaccard_distance(set(instruction), set(response))
    return similarity < threshold


def get_minhash(text, num_perm=128, n_gram=5):
    m = MinHash(num_perm=num_perm)
    tokens = tokenizer.encode(text, add_special_tokens=False)
    # character n-grams over token ids
    shingles = [tuple(tokens[i : i + n_gram]) for i in range(len(tokens) - n_gram + 1)]
    for shingle in shingles:
        m.update(str(shingle).encode("utf-8"))
    return m

from tqdm import tqdm
def dedup_dataset(dataset, threshold=0.8, num_perm=128, n_gram=5):
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    keep_indices = []

    for idx, sample in enumerate(tqdm(dataset, desc="Deduplicating", unit="samples")):
        combined = sample["instruction"] + " " + sample["response"]
        m = get_minhash(combined, num_perm, n_gram)

        if not lsh.query(m):
            lsh.insert(str(idx), m)
            keep_indices.append(idx)

    deduped = dataset.select(keep_indices)
    print(f"Original : {len(dataset)} samples")
    print(f"Deduped  : {len(deduped)} samples")
    print(f"Removed  : {len(dataset) - len(deduped)} samples")
    return deduped

# filter on word counts
filtered_ds = combined_dataset.filter(wc_filter_function)\
    .filter(tokenizer_filter_function)\
        .filter(filter_english)\
            .filter(filter_same_inst_resp_pairs)

# dedup dataset
deduped_dataset = dedup_dataset(filtered_ds)

deduped_dataset.save_to_disk("/home/siddhesh/Documents/repos/llm-and-vlm-projects/datasets/processed/bitext_multiwoz_sft_dataset"   )