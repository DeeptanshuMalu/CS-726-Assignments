import torch
import torch.nn as nn
import warnings

from jaxtyping import Bool, Float, Int
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List

from collections import defaultdict

warnings.filterwarnings("ignore")

class TrieNode:
    def __init__(self):
        self.children = defaultdict(TrieNode)
        self.is_end = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, tok_list):
        node = self.root
        for tok in tok_list:
            node = node.children[tok]
        node.is_end = True

class ConstrainedTextGenerator:
    def __init__(
        self, 
        model: AutoModelForCausalLM, 
        tokenizer: AutoTokenizer, 
        eos_id: int, 
        max_output_len: int = 10,
    ) -> None:
        '''
            Initialize the ConstrainedTextGenerator class.
            
            model: LLM
            tokenizer: LLM's tokenizer.
            eos_id: End-of-sequence token id 
            max_output_len: Maximum number of tokens to be generated.
            
            Do not edit.
        '''
        self.model = model
        
        self.max_output_len = max_output_len
        self.eos_token_id = eos_id
        
        self.tokenizer = tokenizer

    def __call__(
        self, input_ids: Int[torch.Tensor, "batch in_seq_len"], word_list: list
    ) -> Int[torch.Tensor, "batch out_seq_len"]:
        '''
            Implement Word-Constrained decoding technique. (refer assignment document for more details)
            
            `word_list`: contains bag of words for the particular example

            - batch size is always 1; no need to handle generation of multiple sequences simultaneously.
            - stop decoding when: 
                - the end-of-sequence (EOS) token is generated `self.eos_token_id`
                - the predefined maximum number of tokens is reached `self.max_output_len`
            
            Return an integer tensor containing only the generated tokens (excluding input tokens).
            
            Input:
                input_ids: tensor of shape (1, P)
            Returns:
                tensor of shape (T,), where T <= self.max_output_len
        '''    
        # TODO:

        # https://arxiv.org/abs/2010.00904
        input_len = input_ids.shape[1]

        trie = Trie()
        for word in word_list:
            trie.insert(self.tokenizer.encode(word, add_special_tokens=False))
        trie.insert([self.eos_token_id])
        
        node = trie.root
        for _ in range(self.max_output_len):
            last_logits = self.model(input_ids).logits[:, -1, :]
            probs = torch.softmax(last_logits, dim=-1)
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            sorted_probs = sorted_probs.squeeze(0)
            sorted_indices = sorted_indices.squeeze(0)

            next_token_id = None
            for token_id in sorted_indices:
                if token_id.item() in node.children:
                    next_token_id = token_id
                    node = node.children[token_id.item()]

                    # if node.is_end: # is_end | None
                    #     node = trie.root
                    if node.children == {}:
                        node = trie.root
                    break
                
            input_ids = torch.cat([input_ids, next_token_id.reshape(1, 1)], dim=-1)
            if next_token_id.item() == self.eos_token_id:
                break

        return input_ids.reshape(-1)[input_len:]
