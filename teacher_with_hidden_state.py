import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np # If you saved as .npy
import warnings

class Teacher_logits(Dataset):
    def __init__(self, top_k_path='./logits.pt',target_path='./target.pt', hidden_state_path = "./hidden_state.pt",k = 100, teacher_vocab_size=128256, max_new_tokens = 100, default_value = -np.inf, device = 0):
        warnings.filterwarnings("ignore", message="You are using torch.load with weights_only=False*")
        torch.set_warn_always(False)
        self.device = f"cuda:{device}" if torch.cuda.is_available() else "cpu"
        self.topk = torch.load(top_k_path,  map_location="cpu")
        self.k_value = k
        self.teacher_vocab_size = teacher_vocab_size
        self.max_seq_len = max_new_tokens
        self.default_value = default_value
        self.target = torch.load(target_path,  map_location="cpu")
        self.hidden_states = torch.load(hidden_state_path,  map_location="cpu")
        self.prompts = list(self.hidden_states.keys())
        # self.target = torch.lo
        
    def __len__(self):
        return len(self.prompts)

    def _reconstruct_teacher_logits_single(self, top_k_values, top_k_indices):
        """
        Reconstructs full logits for a single example from its Top-K values and indices.
        Args:
            top_k_values: Tensor of shape (seq_len, K)
            top_k_indices: Tensor of shape (seq_len, K)
        Returns:
            Tensor of shape (seq_len, teacher_vocab_size)
        """
        batch_len = top_k_values.size(0)
        seq_len = top_k_values.size(1) # Get seq_len from the actual data
        # Initialize full logits tensor with the fill_value
        # Device handling: reconstruction happens on CPU here. Batch will be moved to GPU later.
        full_logits = torch.full((batch_len,seq_len, self.teacher_vocab_size),
                                 self.default_value,
                                 dtype=top_k_values.dtype, device=self.device)

        # Use scatter_ to place the top_k_values at their respective indices
        # top_k_indices shape is (seq_len, K)
        # full_logits shape is (seq_len, vocab_size)
        # dim=1 is the vocab dimension
        full_logits.scatter_(dim=-1, index=top_k_indices, src=top_k_values)
        return full_logits

    def get_logit(self, prompt):
        data = self.topk.get(prompt)

        teacher_top_k_values = data[0].to(torch.float16).to(self.device) # Ensure dtype
        teacher_top_k_indices = data[1].to(torch.long).to(self.device) # Ensure dtype

        # current_seq_len = teacher_top_k_values.shape[0]
        # 3. Reconstruct full teacher logits for this single example
        reconstructed_teacher_logits = self._reconstruct_teacher_logits_single(
            teacher_top_k_values,
            teacher_top_k_indices
        )
        del teacher_top_k_values
        del teacher_top_k_indices
        return reconstructed_teacher_logits
    
    def get_logits_batch(self, batch):
        logit_list = []
        for prompt in batch:
            logit_list.append(self.get_logit(prompt))
        return logit_list

    def get_hidden_state(self, prompt):
        data = self.hidden_states.get(prompt)

        data = data.to(self.device)
        return data
    
    def get_hidden_state_batch(self, batch):
        hidden_states = []
        for prompt in batch:
            hidden_states.append(self.get_hidden_state(prompt))
        return hidden_states


    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        target = self.target[prompt]['target']
        task = self.target[prompt]['task']
        return {'prompt':prompt, 'target':target, 'task':task}
        data = self.dataset.get(prompt)

        teacher_top_k_values = data[0].to(torch.float16).to(self.device) # Ensure dtype
        teacher_top_k_indices = data[1].to(torch.long).to(self.device) # Ensure dtype

        # current_seq_len = teacher_top_k_values.shape[0]
        # 3. Reconstruct full teacher logits for this single example
        reconstructed_teacher_logits = self._reconstruct_teacher_logits_single(
            teacher_top_k_values,
            teacher_top_k_indices
        )

        return {
            "prompt": prompt,
            "reconstructed_teacher_logits": reconstructed_teacher_logits,
            # "original_idx": idx # Optional
        }