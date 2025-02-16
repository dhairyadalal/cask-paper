from nltk.util import ngrams, everygrams
import random 
import numpy as np 
from dataclasses import dataclass
from transformers import AutoTokenizer, BatchEncoding
from ast import literal_eval


@dataclass
class SimpleCollator:
    tokenizer: AutoTokenizer
    config: dict 
    
    def __call__(self, examples: list) -> dict:
        batch = BatchEncoding(
            {
                k: [examples[i][k] for i in range(len(examples))]
                for k, v in examples[0].items()
            }
        )

        encoded_inputs = self.tokenizer(
            batch[self.config["input_column"]], 
            max_length = 512, 
            padding=True, 
            truncation=True,
            return_tensors="pt"
        )

        encoded_targets = self.tokenizer(
            batch[self.config["output_column"]], max_length = 512, padding=True, truncation=True,
            return_tensors="pt"
        )
        encoded_inputs["labels"] = encoded_targets["input_ids"]

        return encoded_inputs
    

@dataclass
class ConceptCollator:
    tokenizer: AutoTokenizer
    
    def __call__(self, examples: list) -> dict:


        batch = BatchEncoding(
            {
                k: [examples[i][k] for i in range(len(examples))]
                for k, v in examples[0].items()
            }
        )
        
        batch_inputs, batch_labels = [], []
        
        for i, ents in enumerate(batch["entities"]):

            num_sentinels = random.randint(1, len(ents))
            mask_candidates = random.sample(ents, num_sentinels)
            sentinals =[f"<extra_id_{i}>" for i in np.arange(num_sentinels)]
            mask_candidates = random.sample(ents, num_sentinels-1)

            sent = batch["input"][i]

            label = []
            for i, cand in enumerate(mask_candidates):
                sent = sent.replace(cand, sentinals[i])
                label.append(f"{sentinals[i]}{cand}")
            label.append(sentinals[-1])
            label = "".join(label).strip()
            batch_inputs.append(sent)
            batch_labels.append(label)

        input_ids = self.tokenizer(
            batch_inputs, padding=True, truncation=True, return_tensors="pt",
        )["input_ids"]

        labels = self.tokenizer(
            batch_labels, padding=True, truncation=True, return_tensors="pt",
        )["input_ids"]
        return {"input_ids": input_ids, "labels": labels}



@dataclass
class RandomMLMCollator:

    tokenizer: AutoTokenizer
    noise_ratio: float = 0.15
    max_length: int = 512

    def random_mask_input(self, input_sent: str) -> tuple:
        toks = input_sent.split()
        num_sentinels = round(len(toks) * self.noise_ratio) + 1
        sentinals =[f"<extra_id_{i}>" for i in np.arange(num_sentinels)]

        mask_candidates = random.sample(
            [
                " ".join(gram) for gram in list(everygrams(toks, 1,3))
                if len(" ".join(gram)) > 2
            ],
            num_sentinels - 1
        )

        labels = []
        for i, cand in enumerate(mask_candidates):
            input_sent = input_sent.replace(cand, sentinals[i]) 
            labels.append(f"{sentinals[i]}{cand}")

        labels.append(sentinals[-1])
        labels = "".join(labels)
        return input_sent, labels


    def __call__(self, examples: list):
        
        batch = BatchEncoding(
            {
                k: [examples[i][k] for i in range(len(examples))]
                for k, v in examples[0].items()
            }
        )
        
        batch_inputs, batch_labels = [], []

        for ex in batch["input"]:
            input_sent, labels = self.random_mask_input(ex)
            batch_inputs.append(input_sent)
            batch_labels.append(labels)
        
        input_ids = self.tokenizer(
            batch_inputs,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )["input_ids"]

        labels = self.tokenizer(
            batch_labels,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )["input_ids"]

        return {"input_ids": input_ids, "labels": labels}

