"""Export Gal-DINO (pure-PyTorch grid_sample deformable attention + in-graph PostProcess top-k)
to ONNX with the TorchScript exporter, static [1, 3, 800, 800] input, then check it against torch
in ONNX Runtime.

  .venv/bin/python export_onnx.py            -> export/galdino.onnx
"""

import sys
from pathlib import Path

import numpy as np
import onnx
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import galdino_common as gc  # noqa: E402

OUT = HERE / "export"


def main():
    OUT.mkdir(exist_ok=True)
    gc.use_pytorch_msda()
    model, _ = gc.build_model()
    wrapper = gc.ExportWrapper(model).eval()
    oh, ow = gc.out_size(450, 450)
    path = gc.DATA / "test" / "495928_1.png"
    x = gc.preprocess_pil(path)[None].cuda()
    assert tuple(x.shape) == (1, 3, oh, ow)

    onnx_path = OUT / ("galdino.onnx" if gc.MIN_SIZE == 800 else f"galdino_{gc.MIN_SIZE}.onnx")
    with torch.no_grad():
        ref = [t.cpu().numpy() for t in wrapper(x)]
        torch.onnx.export(wrapper, (x,), str(onnx_path), dynamo=False, opset_version=17,
                          input_names=["image"], output_names=["scores", "labels", "boxes", "kpts"],
                          do_constant_folding=True)
    m = onnx.load(str(onnx_path))
    onnx.checker.check_model(m)
    try:
        import onnxsim

        m2, ok = onnxsim.simplify(m)
        if ok:
            onnx.save(m2, str(onnx_path))
            print("onnxsim: ok")
    except Exception as e:  # simplification is optional
        print("onnxsim skipped:", e)
    ops = {}
    for n in onnx.load(str(onnx_path)).graph.node:
        ops[n.op_type] = ops.get(n.op_type, 0) + 1
    print("ops:", dict(sorted(ops.items(), key=lambda kv: -kv[1])))

    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CUDAExecutionProvider",
                                                           "CPUExecutionProvider"])
    got = sess.run(None, {"image": x.cpu().numpy()})
    keep = ref[0][0] > 0.05
    print("ORT vs torch: score max abs", np.abs(got[0] - ref[0]).max(),
          "| top-k index match", (got[1] == ref[1]).mean(),
          "| boxes (score>0.05) max abs", np.abs(got[2][0][keep] - ref[2][0][keep]).max(),
          "| kpts (score>0.05) max abs", np.abs(got[3][0][keep] - ref[3][0][keep]).max())
    print("wrote", onnx_path)


if __name__ == "__main__":
    main()
