#!/usr/bin/env python3
"""Convert Photos concept TFLite + public face ONNX/PB graphs to RK3566 RKNN.

Mean/std is NOT baked into the RKNN: npu_server feeds pre-normalized NCHW
float32 using npu_model_conf.json. Official Vivante .nb weights cannot be
converted; concept uses the official x86 TFLite, face uses public models
that match the npu.Model ABI (512/320 detect, 112x112 / 128-d feature).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import types


def copy_onnx_bundle(src: str, dst: str) -> None:
    """Copy an ONNX file plus same-directory external weight blobs."""
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    shutil.copy2(src, dst)
    src_dir = os.path.dirname(os.path.abspath(src))
    dst_dir = os.path.dirname(os.path.abspath(dst))
    for name in os.listdir(src_dir):
        if name.endswith(".data") or name.endswith(".onnx.data"):
            shutil.copy2(os.path.join(src_dir, name), os.path.join(dst_dir, name))
            print("copied sidecar", name)


def apply_mapping_shim():
    import numpy as np
    import onnx

    if hasattr(onnx, "mapping"):
        return
    m = types.SimpleNamespace()
    m.TENSOR_TYPE_TO_NP_TYPE = {
        1: np.dtype("float32"),
        2: np.dtype("uint8"),
        3: np.dtype("int8"),
        4: np.dtype("uint16"),
        5: np.dtype("int16"),
        6: np.dtype("int32"),
        7: np.dtype("int64"),
        9: np.dtype("bool"),
        10: np.dtype("float16"),
        11: np.dtype("float64"),
        12: np.dtype("uint32"),
        13: np.dtype("uint64"),
        14: np.dtype("complex64"),
        15: np.dtype("complex128"),
    }
    m.NP_TYPE_TO_TENSOR_TYPE = {v: k for k, v in m.TENSOR_TYPE_TO_NP_TYPE.items()}
    m.TENSOR_TYPE_TO_STORAGE_TENSOR_TYPE = m.TENSOR_TYPE_TO_NP_TYPE
    onnx.mapping = m
    print("onnx.mapping shim injected")


def rewrite_input_nchw(onnx_path: str) -> None:
    import onnx
    from onnx import helper

    m = onnx.load(onnx_path)
    if not m.graph.input:
        return
    # Freeze leftover training flags (MobileFaceNet_TF emits phase_train).
    import numpy as np
    from onnx import numpy_helper

    keep = []
    for inp in list(m.graph.input):
        name = inp.name.lower()
        if "phase" in name or name.endswith("train") or "train:" in name:
            init = numpy_helper.from_array(np.array(False), name=inp.name)
            m.graph.initializer.append(init)
            print("froze aux input", inp.name, "-> False")
            continue
        keep.append(inp)
    if keep:
        del m.graph.input[:]
        m.graph.input.extend(keep)
    inp = m.graph.input[0]
    dims = [d.dim_value for d in inp.type.tensor_type.shape.dim]
    if dims and dims[0] in (0, -1):
        inp.type.tensor_type.shape.dim[0].dim_value = 1
        dims[0] = 1
        print("batch dim forced to 1")
    print("onnx ir", m.ir_version, "inputs", [(i.name, [d.dim_value for d in i.type.tensor_type.shape.dim]) for i in m.graph.input])
    print("onnx outputs", [(o.name, [d.dim_value for d in o.type.tensor_type.shape.dim]) for o in m.graph.output])
    if len(dims) == 4 and dims[-1] in (1, 3) and dims[1] not in (1, 3):
        orig = inp.name
        nchw = orig + "_nchw"
        inp.name = nchw
        nchw_dims = [dims[0] or 1, dims[3], dims[1], dims[2]]
        for i, v in enumerate(nchw_dims):
            inp.type.tensor_type.shape.dim[i].dim_value = v
        m.graph.node.insert(0, helper.make_node("Transpose", [nchw], [orig], perm=[0, 2, 3, 1], name="nchw_to_nhwc"))
        print("input rewritten to NCHW", nchw_dims)
    onnx.save(m, onnx_path)


def smoke_onnx(onnx_path: str) -> None:
    import numpy as np
    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    feed = {}
    for inp in sess.get_inputs():
        shape = [d if isinstance(d, int) and d > 0 else 1 for d in (inp.shape or [])]
        typ = (inp.type or "").lower()
        if "bool" in typ:
            feed[inp.name] = np.array(False)
        elif "int64" in typ:
            feed[inp.name] = np.zeros(shape or [1], dtype=np.int64)
        else:
            if not shape:
                shape = [1]
            feed[inp.name] = np.random.rand(*shape).astype(np.float32)
    out = sess.run(None, feed)
    print("smoke", os.path.basename(onnx_path), [o.shape for o in out], "finite", bool(np.isfinite(out[0]).all()))
    del sess


def maybe_smoke(onnx_path: str) -> None:
    try:
        smoke_onnx(onnx_path)
    except Exception as exc:
        print("smoke warning (continuing):", exc)


def tflite_to_onnx(tflite: str, onnx_path: str) -> None:
    r = subprocess.run(
        [sys.executable, "-m", "tf2onnx.convert", "--tflite", tflite, "--output", onnx_path, "--opset", "13"],
        capture_output=True,
        text=True,
    )
    print(r.stdout[-2000:])
    print(r.stderr[-2000:])
    if r.returncode != 0 or not os.path.exists(onnx_path):
        raise SystemExit(f"tf2onnx tflite failed rc={r.returncode}")


def graphdef_to_onnx(pb: str, onnx_path: str) -> None:
    import tensorflow as tf

    gd = tf.compat.v1.GraphDef()
    with open(pb, "rb") as f:
        gd.ParseFromString(f.read())
    names = [n.name for n in gd.node]
    placeholders = [n.name for n in gd.node if n.op == "Placeholder"]
    print("graphdef placeholders", placeholders, "nodes", len(names))
    cmd = [sys.executable, "-m", "tf2onnx.convert", "--graphdef", pb, "--output", onnx_path, "--opset", "13"]
    # sirius-ai MobileFaceNet_TF: img_inputs -> embeddings. Ignore phase_train.
    if "img_inputs" in names:
        cmd += ["--inputs", "img_inputs:0", "--inputs-as-nchw", "img_inputs:0"]
    elif placeholders:
        inputs = [p if ":" in p else p + ":0" for p in placeholders if "phase" not in p.lower()]
        if inputs:
            cmd += ["--inputs", ",".join(inputs)]
    for cand in ("embeddings:0", "embeddings", "output:0", "output", "Bottleneck_BatchNorm:0"):
        base = cand.split(":")[0]
        if base in names or cand in names:
            cmd += ["--outputs", cand if ":" in cand else cand + ":0"]
            break
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout[-2000:])
    print(r.stderr[-2000:])
    if r.returncode != 0 or not os.path.exists(onnx_path):
        raise SystemExit(f"tf2onnx graphdef failed rc={r.returncode}")


def ensure_opset(onnx_path: str, target: int = 13) -> None:
    import onnx
    from onnx import version_converter

    m = onnx.load(onnx_path)
    opsets = [o.version for o in m.opset_import if o.domain in ("", "ai.onnx")]
    current = max(opsets) if opsets else 0
    if current <= 19:
        return
    print(f"downgrading onnx opset {current} -> {target}")
    converted = version_converter.convert_version(m, target)
    onnx.save(converted, onnx_path)


def export_rknn(onnx_path: str, rknn_path: str, platform: str, inputs=None, input_size_list=None) -> None:
    apply_mapping_shim()
    ensure_opset(onnx_path)
    from rknn.api import RKNN

    rknn = RKNN(verbose=True)
    rknn.config(target_platform=platform)
    kwargs = {"model": onnx_path}
    if inputs:
        kwargs["inputs"] = inputs
        if input_size_list:
            kwargs["input_size_list"] = input_size_list
    try:
        ret = rknn.load_onnx(**kwargs)
    except Exception as exc:
        rknn.release()
        raise SystemExit(f"load_onnx exception: {exc}") from exc
    print("load_onnx", onnx_path, "->", ret)
    if ret != 0:
        rknn.release()
        raise SystemExit(f"load_onnx failed: {onnx_path}")
    ret = rknn.build(do_quantization=False)
    print("build ->", ret)
    if ret != 0:
        rknn.release()
        raise SystemExit(f"build failed: {onnx_path}")
    os.makedirs(os.path.dirname(rknn_path) or ".", exist_ok=True)
    ret = rknn.export_rknn(rknn_path)
    print("export ->", ret, rknn_path)
    if ret != 0:
        rknn.release()
        raise SystemExit(f"export failed: {rknn_path}")
    rknn.release()
    print("RKNN OK", rknn_path, os.path.getsize(rknn_path))


def input_hw(onnx_path: str) -> tuple[int, int]:
    import onnx

    m = onnx.load(onnx_path)
    dims = [d.dim_value for d in m.graph.input[0].type.tensor_type.shape.dim]
    if len(dims) == 4 and dims[1] in (1, 3):
        return int(dims[3] or 112), int(dims[2] or 112)  # W, H
    if len(dims) == 4:
        return int(dims[2] or 112), int(dims[1] or 112)
    return 112, 112


def write_conf(path: str, detect_w: int, detect_h: int, detect_kind: str, feat_w: int, feat_h: int) -> None:
    retina = detect_kind == "retinaface"
    ultra = detect_kind == "ultraface"
    conf = {
        "detection": {
            "reorder": bool(retina),
            "rgb_mean": [104.0, 117.0, 123.0] if retina else [127.0, 127.0, 127.0],
            "rgb_scale": 1.0 if retina else 0.0078125,
            "net_size": [detect_w, detect_h, 3],
            "result_thres": 0.5,
            "feature_size": {"value": 245760, "_comment": "128 * 128 * 15"},
            "norm_scale": -1,
            "iou_thres": 0.5,
            "fit": "letterbox" if retina else "stretch",
            "decode": "retinaface" if retina else ("ultraface" if ultra else "auto"),
        },
        "feature": {
            "reorder": False,
            "rgb_mean": [127.5, 127.5, 127.5],
            "rgb_scale": 0.0078125,
            "net_size": [feat_w, feat_h, 3],
            "result_thres": -1,
            "feature_size": 128,
            "norm_scale": 100,
            "iou_thres": -1,
            "fit": "stretch",
        },
        "concept": {
            "reorder": False,
            "rgb_mean": [127.5, 127.5, 127.5],
            "rgb_scale": 0.00784313725,
            "net_size": [395, 395, 3],
            "result_thres": -1,
            "feature_size": 335,
            "norm_scale": -1,
            "iou_thres": -1,
            "fit": "letterbox",
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(conf, f, indent=2)
        f.write("\n")
    print("wrote", path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--platform", default="rk3566")
    ap.add_argument("--concept-tflite")
    ap.add_argument("--detection-onnx")
    ap.add_argument("--feature-onnx")
    ap.add_argument("--feature-pb")
    ap.add_argument("--skip-concept", action="store_true")
    ap.add_argument("--skip-detection", action="store_true")
    ap.add_argument("--skip-feature", action="store_true")
    args = ap.parse_args()

    apply_mapping_shim()
    out = args.out_dir
    os.makedirs(out, exist_ok=True)
    work = os.path.join(out, "_onnx")
    os.makedirs(work, exist_ok=True)

    detect_w, detect_h, detect_kind = 512, 512, "auto"
    feat_w, feat_h = 112, 112

    if not args.skip_concept:
        if not args.concept_tflite or not os.path.isfile(args.concept_tflite):
            raise SystemExit("concept tflite missing")
        onnx_path = os.path.join(work, "concept.onnx")
        tflite_to_onnx(args.concept_tflite, onnx_path)
        rewrite_input_nchw(onnx_path)
        maybe_smoke(onnx_path)
        export_rknn(onnx_path, os.path.join(out, "concept_network.rknn"), args.platform)

    if not args.skip_detection:
        if not args.detection_onnx or not os.path.isfile(args.detection_onnx):
            raise SystemExit("detection onnx missing")
        onnx_path = os.path.join(work, "detection.onnx")
        if os.path.abspath(args.detection_onnx) != os.path.abspath(onnx_path):
            copy_onnx_bundle(args.detection_onnx, onnx_path)
        rewrite_input_nchw(onnx_path)
        maybe_smoke(onnx_path)
        detect_w, detect_h = input_hw(onnx_path)
        name = os.path.basename(args.detection_onnx).lower()
        if "retina" in name:
            detect_kind = "retinaface"
        elif "rfb" in name or "ultra" in name:
            detect_kind = "ultraface"
        export_rknn(onnx_path, os.path.join(out, "detection_network.rknn"), args.platform)

    if not args.skip_feature:
        onnx_path = os.path.join(work, "feature.onnx")
        candidates = []
        if args.feature_pb and os.path.isfile(args.feature_pb):
            candidates.append(("pb", args.feature_pb))
        if args.feature_onnx and os.path.isfile(args.feature_onnx):
            candidates.append(("onnx", args.feature_onnx))
        if not candidates:
            raise SystemExit("feature onnx/pb missing")
        last_err = None
        exported = False
        for kind, src in candidates:
            try:
                print("trying feature", kind, src)
                if kind == "pb":
                    graphdef_to_onnx(src, onnx_path)
                else:
                    copy_onnx_bundle(src, onnx_path)
                rewrite_input_nchw(onnx_path)
                maybe_smoke(onnx_path)
                feat_w, feat_h = input_hw(onnx_path)
                extra = {}
                # Qualcomm export is a 2-image siamese graph; keep the first tower.
                import onnx as _onnx

                m = _onnx.load(onnx_path)
                if len(m.graph.input) > 1 and kind == "onnx":
                    extra = {
                        "inputs": [m.graph.input[0].name],
                        "input_size_list": [[1, 3, feat_h, feat_w]],
                    }
                export_rknn(onnx_path, os.path.join(out, "feature_network.rknn"), args.platform, **extra)
                exported = True
                break
            except SystemExit as exc:
                last_err = exc
                print("feature candidate failed:", exc)
        if not exported:
            raise SystemExit(last_err or "feature convert failed")

    write_conf(os.path.join(out, "npu_model_conf.json"), detect_w, detect_h, detect_kind, feat_w, feat_h)
    print("FINAL OK", out)


if __name__ == "__main__":
    main()
