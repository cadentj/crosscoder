from utils import *
from trainer import Trainer
import zstandard as zstd
from nnsight import LanguageModel
import io
import json

device = 'cuda:0'
base_path = "/share/data/datasets/pile/the-eye.eu/public/AI/pile/train"

def load_zst_files(base_path):
    for i in range(30):  # 00 to 29
        filepath = os.path.join(base_path, f"{i:02d}.jsonl.zst")
        with open(filepath, 'rb') as f:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(f) as reader:
                text_stream = io.TextIOWrapper(reader, encoding='utf-8')
                for line in text_stream:
                    if line.strip():  # Skip empty lines
                        doc = json.loads(line)
                        yield doc['text']

def main():
    base_model = LanguageModel("EleutherAI/pythia-70m-deduped", device_map="auto")
    submodule_A = base_model.gpt_neox.layers[3]

    chat_model = LanguageModel("kh4dien/pythia-70m-deduped-gender", device_map="auto")
    submodule_B = chat_model.gpt_neox.layers[3]

    d_model = 512 # Hardcoded for d_resid
    data = load_zst_files(base_path)
    data = iter(data)
    
    default_cfg = {
        "seed": 49,
        "batch_size": 4096,
        "buffer_mult": 256,
        "lr": 5e-5,
        "num_tokens": 400_000_000,
        "l1_coeff": 2,
        "beta1": 0.9,
        "beta2": 0.999,
        "d_in": d_model,
        "dict_size": 2**16,
        "d_model": d_model,
        "seq_len": 1024,
        "enc_dtype": "fp32",
        "model_name": "EleutherAI/pythia-70m-deduped",
        "site": "resid_pre",
        "device": "cuda:0",
        "model_batch_size": 8,
        "log_every": 100,
        "save_every": 30_000,
        "dec_init_norm": 0.08,
        "wandb_project": "crosscoder",
        "wandb_entity": "gradient-features",
    }
    cfg = arg_parse_update_cfg(default_cfg)

    trainer = Trainer(
        data=data, 
        cfg=cfg, 
        model_A=base_model, 
        model_B=chat_model,
        submodule_A=submodule_A,
        submodule_B=submodule_B,
    )
    trainer.train()

if __name__ == "__main__":
    main()