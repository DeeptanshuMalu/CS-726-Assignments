import torch
import torch.nn as nn
import warnings

from jaxtyping import Bool, Float, Int
from transformers import AutoModelForCausalLM
from typing import List

warnings.filterwarnings("ignore")

class MedusaTextGenerator:
    def __init__(
        self, 
        model: AutoModelForCausalLM, 
        decoding_strategy: str, 
        eos_id: int, 
        use_no_medusa_heads: int = 5,
        beam_width: int = 2,
        max_output_len: int = 10,
    ) -> None:
        '''
            Initialize the MedusaTextGenerator class.
            
            model: LLM
            decoding_strategy: str describing the decoding strategy to be used.
            eos_id: End-of-sequence token id 
            use_no_medusa_heads: Number of medusa heads to be used (maximum:5) (denoted as S).
            beam_width: Maximum number of candidates that can be present in the beam (denoted as W).
            max_output_len: Maximum number of tokens to be generated.
            
            Do not edit.
        '''
        self.model = model
        self.decoding_strategy = decoding_strategy
        
        self.max_output_len = max_output_len
        self.eos_token_id = eos_id
        self.beam_width = beam_width
        
        assert use_no_medusa_heads <= 5, "The current medusa model supports at max 5 heads"
        self.no_heads = use_no_medusa_heads + 1
        
        if decoding_strategy == "single-head":
            self.generator_func = self.single_head_decoding
        elif decoding_strategy == "multi-head":
            self.generator_func = self.multi_head_decoding
        
    def __call__(
        self, input_ids: Int[torch.Tensor, "batch in_seq_len"]
    ) -> Int[torch.Tensor, "batch out_seq_len"]:
        '''
            Do not edit.
        '''
        return self.generator_func(input_ids)
                
    def single_head_decoding(
        self,
        input_ids: Float[torch.Tensor, "batch in_seq_len"],
    ) -> Int[torch.Tensor, "batch out_seq_len"]:     
        '''
            Implement Single-head decoding technique. Use only LM head for decoding here (refer assignment document for more details)

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
        input_len = input_ids.shape[1]
        with torch.inference_mode():
            for _ in range(self.max_output_len):
                medusa_logits, outputs, logits = self.model(input_ids, output_orig = True, medusa_forward = True)
                last_logits = logits[:, -1, :]
                probs = torch.softmax(last_logits, dim=-1)
                next_token_id = torch.argmax(probs, dim=-1, keepdim=True)
                input_ids = torch.cat([input_ids, next_token_id], dim=-1)
                if next_token_id.item() == self.eos_token_id:
                    break
        return input_ids.reshape(-1)[input_len:]

    def multi_head_decoding(
        self,
        input_ids: Float[torch.Tensor, "batch in_seq_len"],
    ) -> Int[torch.Tensor, "batch out_seq_len"]:     
        '''
            Implement multi-head decoding technique. (refer assignment document for more details)

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
        
        input_len = input_ids.shape[1]
        output_length = 0
        with torch.inference_mode():
            while output_length < self.max_output_len:
                medusa_logits, outputs, logits = self.model(input_ids, output_orig = True, medusa_forward = True)
                last_logits = logits[0, -1, :]
                last_medusa_logits = medusa_logits[:self.no_heads-1, 0, -1, :]
                candidates = [input_ids[0]]
                scores = torch.zeros(1)
            
                for s in range(self.no_heads):
                    if s==0:
                        log_probs = torch.log_softmax(last_logits, dim=-1)
                    else:
                        log_probs = torch.log_softmax(last_medusa_logits[s - 1], dim=-1)
                        
                    new_candidates = []
                    new_scores = []

                    sorted_probs, sorted_indices = torch.sort(log_probs, descending=True)
                    top_w_probs = sorted_probs[:self.beam_width]
                    top_w_indices = sorted_indices[:self.beam_width]
                    for c in range(len(candidates)):
                        if candidates[c][-1] == self.eos_token_id:
                            new_candidates.append(candidates[c])
                            new_scores.append(scores[c])
                            continue

                        for i in range(self.beam_width):
                            new_score = scores[c] + top_w_probs[i]
                            new_candidate = torch.cat([candidates[c], top_w_indices[i].unsqueeze(0)], dim=-1)
                            new_candidates.append(new_candidate)
                            new_scores.append(new_score)

                    scores, indices = torch.sort(torch.stack(new_scores), descending=True)
                    candidates = [new_candidates[i] for i in indices[:self.beam_width]]
                    scores = scores[:self.beam_width]

                output_length = len(candidates[0]) - input_len
                input_ids = candidates[0].unsqueeze(0)

                if candidates[0][-1] == self.eos_token_id:
                    break

            return input_ids.reshape(-1)[input_len:input_len + output_length]