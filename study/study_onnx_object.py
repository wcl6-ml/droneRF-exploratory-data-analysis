#%%
from pathlib import Path

import onnxruntime as ort


def _find_project_root(marker: str = "pyproject.toml") -> Path:
    current = Path(__file__).resolve().parent
    for parent in [current, *current.parents]:
        if (parent / marker).exists():
            return parent
    raise RuntimeError(f"Could not find project root (looking for {marker})")


root_dir = _find_project_root()
model_path = root_dir / "models" / "model.onnx"
print(f"Using model path: {model_path}")

sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
#%%
print(type(sess))
print(sess.get_inputs())
print(sess.get_outputs())
inp = sess.get_inputs()[0]
print(type(inp))
print(inp.name)
print(inp.shape)
print(inp.type)
# %%
import numpy as np
import onnxruntime as ort

x = np.random.randn(32, 1, 14).astype(np.float32)
sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])

input_name = sess.get_inputs()[0].name
output = sess.run(None, {input_name: x})[0]

print(x.shape)
print(output.shape)
print(output[:3])
# %%
