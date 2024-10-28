import gc
from typing import Iterator

from tqdm import tqdm
from nnsight.envoy import Envoy
from nnsight import LanguageModel

from .utils import *

class Buffer:
    """
    This defines a data buffer, to store a stack of acts across both models that can be used to train the autoencoder. 
    It'll automatically run the model to generate more when it gets halfway empty.
    """

    def __init__(
        self, 
        data: Iterator[str], 
        cfg, 
        model_A: LanguageModel, 
        model_B: LanguageModel,
        submodule_A: Envoy,
        submodule_B: Envoy,
    ):
        # Load buffer and set to device
        self.buffer_size = cfg["batch_size"] * cfg["buffer_mult"]
        self.buffer = torch.zeros(
            (self.buffer_size, 2, cfg["d_model"]),
            dtype=torch.bfloat16,
            requires_grad=False,
        ).to(cfg["device"]) # hardcoding 2 for model diffing

        # Set up class attributes
        self.cfg = cfg
        self.data = data
        self.model_A = model_A
        self.model_B = model_B
        self.submodule_A = submodule_A
        self.submodule_B = submodule_B
        self.normalize = True
        
        estimated_norm_scaling_factor_A = self.estimate_norm_scaling_factor(
            cfg["model_batch_size"], 
            model_A, 
            submodule_A
        )
        estimated_norm_scaling_factor_B = self.estimate_norm_scaling_factor(
            cfg["model_batch_size"], 
            model_B, 
            submodule_B
        )
        
        self.normalisation_factor = torch.tensor(
            [
                estimated_norm_scaling_factor_A,
                estimated_norm_scaling_factor_B,
            ],
            device="cuda:0",
            dtype=torch.float32,
        )

        self.pointer = 0
        self.refresh()

    def get_normalization_factor(self):
        return {
            "A": self.normalisation_factor[0],
            "B": self.normalisation_factor[1],
        }

    @torch.no_grad()
    def estimate_norm_scaling_factor(self, batch_size, model, submodule, n_batches_for_norm_estimate: int = 100):
        # stolen from SAELens https://github.com/jbloomAus/SAELens/blob/6d6eaef343fd72add6e26d4c13307643a62c41bf/sae_lens/training/activations_store.py#L370
        norms_per_batch = []
        for i in tqdm(
            range(n_batches_for_norm_estimate), desc="Estimating norm scaling factor"
        ):
            tokens = self.tokenized_batch(batch_size)

            with model.trace(tokens):
                acts = submodule.output[0].save()

            # TODO: maybe drop BOS here
            norms_per_batch.append(acts.norm(dim=-1).mean().item())
        mean_norm = np.mean(norms_per_batch)
        scaling_factor = np.sqrt(self.cfg["d_model"]) / mean_norm

        return scaling_factor

    def tokenized_batch(self, batch_size: int):
        try:
            texts = [next(self.data) for _ in range(batch_size)]
            return self.model_A.tokenizer(
                texts,
                return_tensors='pt',
                max_length=self.cfg["seq_len"],
                padding=True,
                truncation=True
            )
        except StopIteration:
            raise StopIteration("End of data stream reached")

    @torch.no_grad()
    def refresh(self):
        print("Refreshing the buffer!")
        with torch.autocast("cuda", torch.bfloat16):
            # Reset token pointer to the current pointer position
            token_pointer = self.pointer
            while token_pointer < self.buffer_size:
                inputs = self.tokenized_batch(min(
                    self.buffer_size - token_pointer,
                    self.cfg["model_batch_size"]
                ))
                tokens = inputs["input_ids"]
                attn_mask = inputs["attention_mask"]
                
                with self.model_A.trace(tokens):
                    output_A = self.submodule_A.output[0].save()

                with self.model_B.trace(tokens):
                    output_B = self.submodule_B.output[0].save()

                acts = torch.stack([output_A.value, output_B.value], dim=2)

                # NOTE: There is no BOS token for Pythia, so we don't need to drop it
                # acts = acts[:, 1:, :, :] # Drop BOS token

                assert acts.shape == (tokens.shape[0], tokens.shape[1], 2, self.cfg["d_model"]) # [batch, seq_len, 2, d_model]
                acts = acts[attn_mask != 0]

                # Fix: Calculate remaining space in buffer and limit acts accordingly
                remaining_space = self.buffer_size - token_pointer
                pointer_increment = min(acts.shape[0], remaining_space)
                self.buffer[token_pointer : token_pointer + pointer_increment] = acts[:pointer_increment]
                token_pointer += pointer_increment

        # Reset pointer and randomize buffer
        self.pointer = 0
        self.buffer = self.buffer[
            torch.randperm(self.buffer.shape[0]).to(self.cfg["device"])
        ]

        torch.cuda.empty_cache()
        gc.collect()

    @torch.no_grad()
    def next(self):
        out = self.buffer[self.pointer : self.pointer + self.cfg["batch_size"]].float()
        # out: [batch_size, n_layers, d_model]
        self.pointer += self.cfg["batch_size"]
        if self.pointer > self.buffer.shape[0] // 2 - self.cfg["batch_size"]:
            self.refresh()
        if self.normalize:
            out = out * self.normalisation_factor[None, :, None]
        return out
