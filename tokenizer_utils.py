from tokenizers import ByteLevelBPETokenizer
from transformers import PreTrainedTokenizerFast
from settings import Settings

def build_tokenizer_from_texts(text_samples, vocab_size=Settings.VOCABULARY_SIZE):
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train_from_iterator(
        text_samples,
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=Settings.SPECIAL_TOKENS
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token=Settings.BOS_TOKEN,
        eos_token=Settings.EOS_TOKEN,
        unk_token="<UNK>",
        pad_token=Settings.PAD_TOKEN,
        sep_token="<SEP>"
    )