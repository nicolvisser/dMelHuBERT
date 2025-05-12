from dataclasses import dataclass
from itertools import groupby
from pathlib import Path
from typing import List, Optional

from argparse_dataclass import ArgumentParser
from tokenizers import Encoding, Tokenizer
from tokenizers.decoders import Decoder
from tokenizers.decoders import WordPiece as WordPieceDecoder
from tokenizers.models import WordPiece
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.processors import PostProcessor
from tokenizers.trainers import WordPieceTrainer
from torch.utils.data import Dataset


class WordPieceDecoderWithCTC:
    def __init__(self, blank_token: str = "-"):
        super().__init__()
        self.blank_token = blank_token
        self.word_piece_decoder: WordPieceDecoder = WordPieceDecoder()

    def decode(self, tokens: List[str]) -> str:
        """
        Decode the given list of tokens to a final string.
        First removes consecutive duplicates and any blank tokens, then runs the WordPiece decoder.

        Args:
            tokens (:obj:`List[str]`):
                The list of tokens to decode

        Returns:
            :obj:`str`: The decoded string
        """
        ctc_decoded_tokens = [k for k, g in groupby(tokens) if k != self.blank_token]
        return self.word_piece_decoder.decode(ctc_decoded_tokens)


class CTCPostProcessor:
    def __init__(self, blank_id: int, blank_token: str = "-"):
        super().__init__()
        self.blank_id = blank_id
        self.blank_token = blank_token

    def num_special_tokens_to_add(self, is_pair):
        """
        Return the number of special tokens that would be added for single/pair sentences.

        Args:
            is_pair (:obj:`bool`):
                Whether the input would be a pair of sequences

        Returns:
            :obj:`int`: The number of tokens to add
        """
        return 1

    def process(
        self,
        encoding: Encoding,
        pair: Optional[Encoding] = None,
        add_special_tokens: bool = True,
    ):
        """
        Post-process the given encodings, generating the final one.
        This looks for repeated tokens and puts a blank token between them.

        Args:
            encoding (:class:`~tokenizers.Encoding`):
                The encoding for the first sequence

            pair (:class:`~tokenizers.Encoding`, `optional`):
                The encoding for the pair sequence

            add_special_tokens (:obj:`bool`):
                Whether to add the special tokens. Not used in this post-processor.
                CTC always adds the special tokens.

        Return:
            :class:`~tokenizers.Encoding`: The final encoding
        """
        ids = []
        tokens = []
        attention_mask = []
        special_tokens_mask = []
        type_ids = []
        word_ids = []
        offsets = []
        sequence_ids = []

        prev_id = None
        for idx, (id, tk, am, stm, ti, wi, of, si) in enumerate(
            zip(
                encoding.ids,
                encoding.tokens,
                encoding.attention_mask,
                encoding.special_tokens_mask,
                encoding.type_ids,
                encoding.word_ids,
                encoding.offsets,
                encoding.sequence_ids,
            )
        ):
            if idx > 0 and id == prev_id:
                # make provisions for the blank token
                ids.append(self.blank_id)
                tokens.append(self.blank_token)
                attention_mask.append(1)
                special_tokens_mask.append(1)
                type_ids.append(0)
                word_ids.append(None)
                offsets.append((idx, idx))
                sequence_ids.append(si)

            # add the current token
            ids.append(id)
            tokens.append(tk)
            attention_mask.append(am)
            special_tokens_mask.append(stm)
            type_ids.append(ti)
            word_ids.append(wi)
            offsets.append(of)
            sequence_ids.append(si)

            # remember the current token id
            prev_id = id


@dataclass
class TrainTokenizerArgs:
    text_dir: str = "/mnt/wsl/data/transcriptions/LibriSpeech"
    vocab_size: int = 512
    output_path: str = "./tokenizer.json"
    pattern: str = "**/*.txt"


def train_tokenizer(args: TrainTokenizerArgs):
    output_path = Path(args.output_path)
    assert output_path.suffix == ".json"
    assert args.vocab_size > 0

    # build tokenizer
    special_tokens = ["-"]
    tokenizer = Tokenizer(model=WordPiece())
    tokenizer.add_special_tokens(special_tokens)
    tokenizer.pre_tokenizer = WhitespaceSplit()

    # build trainer
    trainer = WordPieceTrainer(
        vocab_size=args.vocab_size, special_tokens=special_tokens
    )

    # define dataset
    dataset = StringsDataset(
        dir=args.text_dir,
        pattern=args.pattern,
        return_paths=False,
    )

    # train tokenizer
    print("Training tokenizer...")
    tokenizer.train_from_iterator(dataset, trainer=trainer, length=len(dataset))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output_path))
    print(f"Tokenizer saved to {output_path}")


class StringsDataset(Dataset):
    def __init__(
        self,
        dir: str,
        pattern: str = "**/*.txt",
        return_paths: bool = False,
    ):
        super().__init__()
        self.return_paths = return_paths
        self.strings_paths = sorted(list(Path(dir).glob(pattern)))
        print(f"Found {len(self.strings_paths)} files in the dataset.")

    def __len__(self):
        return len(self.strings_paths)

    def __getitem__(self, idx):
        path: Path = self.strings_paths[idx]
        string = path.read_text()
        if self.return_paths:
            return string, path
        return string


if __name__ == "__main__":
    parser = ArgumentParser(TrainTokenizerArgs)
    args = parser.parse_args()
    train_tokenizer(args)
