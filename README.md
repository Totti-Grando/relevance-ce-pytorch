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

## Get the weights

`model.safetensors` and `head.pt` are stored via Git LFS, so a plain `git clone` without git-lfs
leaves them as ~130-byte pointer files. Two ways to get the real bytes:

**A. With git-lfs installed:**
```bash
git clone https://github.com/Totti-Grando/relevance-ce-pytorch.git
cd relevance-ce-pytorch && git lfs pull
```

**B. No git-lfs (download the release assets — plain HTTPS, works with curl or Python):**
```bash
git clone https://github.com/Totti-Grando/relevance-ce-pytorch.git
cd relevance-ce-pytorch
curl -L -o model-150m-all/encoder/model.safetensors \
  https://github.com/Totti-Grando/relevance-ce-pytorch/releases/download/v1/model.safetensors
curl -L -o model-150m-all/head.pt \
  https://github.com/Totti-Grando/relevance-ce-pytorch/releases/download/v1/head.pt
```
Or in pure Python (no curl needed):
```python
import urllib.request
base = "https://github.com/Totti-Grando/relevance-ce-pytorch/releases/download/v1"
urllib.request.urlretrieve(f"{base}/model.safetensors", "model-150m-all/encoder/model.safetensors")
urllib.request.urlretrieve(f"{base}/head.pt", "model-150m-all/head.pt")
```

## Install (Intel GPU)
```bash
pip install torch --index-url https://download.pytorch.org/whl/xpu   # XPU-enabled build
pip install "transformers>=4.51" safetensors
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
