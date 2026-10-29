# relevance-ce-pytorch

PyTorch weights for the relevance cross-encoder (`model-150m-all` — the best checkpoint,
trained on synthetic + HotpotQA). Reconstructed losslessly from the deployed fp32 ONNX and
verified to match ONNX logits to ~1e-5. Use this for backends ONNX Runtime doesn't cover —
e.g. an Intel GPU via `torch.xpu`.

## Layout
```
model-150m-all/
  encoder/            # HF ModernBERT backbone (config + model.safetensors + tokenizer)
  head.pt             # dense/norm/out head + temperature
  meta.json           # labels, utility map, temperature, max_length
common.py             # model definition + load_model/format_pair (copied from the training repo)
```

## Install (Intel GPU)
```bash
pip install torch --index-url https://download.pytorch.org/whl/xpu   # XPU-enabled build
pip install "transformers>=4.51" safetensors
git lfs pull                                                         # fetch the weight files
```

## Run
```python
import torch
from common import load_model, format_pair

print("XPU:", torch.xpu.is_available())          # True on the Intel GPU
m, tok, meta = load_model("model-150m-all")
m = m.to("xpu").eval()

rows = [{"question": "What were Northwind's net flows in Q2 2026?",
         "claim": "Northwind took in $1.2B in Q2 2026.", "q_tags": "", "c_tags": ""}]
q, c = zip(*[format_pair(r) for r in rows])
enc = tok(list(q), list(c), padding=True, truncation="longest_first",
          max_length=meta["max_length"], return_tensors="pt").to("xpu")
with torch.no_grad():
    probs = torch.softmax(m(enc["input_ids"], enc["attention_mask"]) / meta["temperature"], -1)
print([meta["labels"][i] for i in probs.argmax(-1)])   # -> ['DIRECT']
```

Labels and their relevance utilities live in `meta.json`; `DIRECT`/`PARTIAL` are the relevant
classes. If `torch.xpu.is_available()` is `False`, update the Intel GPU driver/runtime (a driver
issue, not a pip package). CPU/CUDA work too — just change `"xpu"` to `"cpu"`/`"cuda"`.
