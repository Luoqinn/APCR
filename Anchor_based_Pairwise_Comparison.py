import time, argparse, datetime, torch, json
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import numpy as np


class APC:
    def __init__(self, model_name, prompt_sequence1_file, prompt_sequence2_file, candidate_size, anchor_idx, device):

        # load LLM
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
        ).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
        )
        self.model.config.pad_token_id = self.model.config.eos_token_id
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # get A/B's token ids
        self.A_token_id = self.tokenizer.encode("A", add_special_tokens=False)
        self.B_token_id = self.tokenizer.encode("B", add_special_tokens=False)

        # load prompts
        self.prompts_sequence1 = self.load_prompt(prompt_sequence1_file)
        self.prompts_sequence2 = self.load_prompt(prompt_sequence2_file)
        assert len(self.prompts_sequence1) == len(self.prompts_sequence2)

        self.candidate_size = candidate_size
        self.anchor_idx = anchor_idx

    def load_prompt(self, prompt_file):
        with open(prompt_file, 'r') as f1:
            datas = json.load(f1)
            formatted_prompts = [
                self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": data}],
                    tokenize=False,
                    add_generation_prompt=True
                ) for data in datas
            ]
        return formatted_prompts

    def get_probs(self, formatted_prompt):
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.model.device)

        with torch.no_grad():
            logits = self.model(inputs.input_ids, attention_mask=inputs.attention_mask).logits

        next_token_logits = logits[0, -1, :]

        probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
        probs_a = probs[self.A_token_id].item()
        probs_b = probs[self.B_token_id].item()

        return probs_a, probs_b

    def anchor_based_pairwise_comparison(self):
        # first round anchor based pairwise comparison for sequence1
        probs_anchor_sequence1 = []
        probs_other_sequence1 = []
        for formatted_prompt in tqdm(self.prompts_sequence1, desc='Predicting sequence1 prompt'):
            probs_a, probs_b = self.get_probs(formatted_prompt)
            probs_anchor_sequence1.append(probs_a)
            probs_other_sequence1.append(probs_b)

        # second round anchor based pairwise comparison for sequence2
        probs_anchor_sequence2 = []
        probs_other_sequence2 = []
        for formatted_prompt in tqdm(self.prompts_sequence2, desc='Predicting sequence2 prompt'):
            probs_a, probs_b = self.get_probs(formatted_prompt)
            probs_anchor_sequence2.append(probs_b)
            probs_other_sequence2.append(probs_a)


        np_probs_anchor_sequence1 = np.array(probs_anchor_sequence1)
        np_probs_other_sequence1 = np.array(probs_other_sequence1)
        np_probs_anchor_sequence2 = np.array(probs_anchor_sequence2)
        np_probs_other_sequence2 = np.array(probs_other_sequence2)


        # calculate preference score
        llm_preference_scores = 0.5 * (
                (np_probs_other_sequence1 - np_probs_anchor_sequence1) +
                (np_probs_other_sequence2 - np_probs_anchor_sequence2)
        )
        llm_preference_scores = llm_preference_scores.reshape(-1, self.candidate_size - 1)
        llm_preference_scores = np.insert(llm_preference_scores, self.anchor_idx, 0, axis=1)
        return llm_preference_scores


