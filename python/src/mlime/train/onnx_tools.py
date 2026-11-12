"""What every ONNX export of the project shares: weights files, quantisation, the tokenizer.

The char LM, route A and the key-level model each write graphs whose
initializers live in one external weights file per graph, tabulated in the
manifest the Rust loader reads. The graphs are quantised one of two ways: the
MatMul weights alone folded to int8 behind a ``DequantizeLinear``
(:func:`quantize_weights_int8`, activations stay float32 so a row's value does
not depend on its batchmates), or dynamic per-channel int8
(:func:`quantize_dynamic_int8`, activations quantised at run time).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnx.external_data_helper

from mlime.train.samples import BaseTokenizer


def external_tensors(model: onnx.ModelProto) -> Iterator[onnx.TensorProto]:
    """Every initializer of *model*, nested subgraphs included, backed by external data."""
    graphs = [model.graph]
    while graphs:
        graph = graphs.pop()
        for tensor in graph.initializer:
            if tensor.data_location == onnx.TensorProto.EXTERNAL:
                yield tensor
        for node in graph.node:
            for attribute in node.attribute:
                if attribute.HasField("g"):
                    graphs.append(attribute.g)
                graphs.extend(attribute.graphs)


def externalize(graph: Path, location: str) -> onnx.ModelProto:
    """Move every initializer of *graph* into the *location* file beside it.

    *location* is relative to the graph's directory. Returns the saved model
    reloaded without its external data (``load_external_data=False`` keeps the
    ``external_data`` entries; a plain ``onnx.load`` folds the bytes back into
    ``raw_data`` and strips them), so the callers read offsets and lengths of
    what was written, not of the in-memory proto.
    """
    (graph.parent / location).unlink(missing_ok=True)  # appends otherwise
    model = onnx.load(str(graph))
    # The file is written back-to-back in initializer order, so a tensor's
    # offset stays a multiple of its element size when larger elements go
    # first -- the Rust side hands the runtime raw pointers into the mapped
    # file, which must be aligned for their dtype.
    initializers = sorted(
        model.graph.initializer,
        key=lambda tensor: -onnx.helper.tensor_dtype_to_np_dtype(tensor.data_type).itemsize,
    )
    model.graph.ClearField("initializer")
    model.graph.initializer.extend(initializers)
    onnx.external_data_helper.convert_model_to_external_data(
        model,
        all_tensors_to_one_file=True,
        location=location,
        size_threshold=0,
        convert_attribute=False,
    )
    onnx.save_model(model, str(graph))
    return onnx.load(str(graph), load_external_data=False)


def weights_table(model: onnx.ModelProto) -> dict[str, dict[str, Any]]:
    """The manifest's tensor table of a saved externalized model, keyed by name."""
    table = {}
    for tensor in external_tensors(model):
        entries = {entry.key: entry.value for entry in tensor.external_data}
        table[tensor.name] = {
            "name": tensor.name,
            "dtype": np.dtype(onnx.helper.tensor_dtype_to_np_dtype(tensor.data_type)).name,
            "shape": list(tensor.dims),
            "offset": int(entries.get("offset", "0")),
            "length": int(entries["length"]),
        }
    return table


def quantize_weights_int8(graph: Path) -> None:
    """Rewrite *graph* in place keeping only the MatMul weights int8.

    Each weight folds into a per-output-channel int8 constant plus a
    ``DequantizeLinear`` that rebuilds it float32 ahead of the matmul, so
    activations stay float32 end to end and no row's value depends on its
    batchmates. Full int8 range applies: nothing here reaches a u8 x s8
    kernel, so the seven-bit guard of the dynamic path is unneeded.
    """
    model = onnx.load(str(graph))
    initializers = {tensor.name: tensor for tensor in model.graph.initializer}
    added: list[onnx.TensorProto] = []
    dropped: set[str] = set()
    dequantized: dict[str, str] = {}  # weight name -> dequantized tensor name
    nodes: list[onnx.NodeProto] = []
    for node in model.graph.node:
        weight_input = -1
        if node.op_type == "MatMul":
            for index in (0, 1):
                if node.input[index] in initializers:
                    weight_input = index
                    break
        if weight_input >= 0:
            wname = node.input[weight_input]
            if wname not in dequantized:
                weight = onnx.numpy_helper.to_array(initializers[wname]).astype(np.float32)
                # Channels are the matmul's output features: the last axis of
                # a [K, N] weight, or the first of [N, K].
                axis = weight.ndim - 1 if weight_input == 1 else 0
                bound = np.abs(weight).max(axis=axis ^ 1)
                scale = (bound / 127.0).astype(np.float32)
                scale[scale == 0.0] = 1.0
                shape = [1] * weight.ndim
                shape[axis] = weight.shape[axis]
                quantized_weight = np.clip(
                    np.round(weight / scale.reshape(shape)), -127, 127
                ).astype(np.int8)
                added += [
                    onnx.numpy_helper.from_array(quantized_weight, f"{wname}_int8"),
                    onnx.numpy_helper.from_array(scale, f"{wname}_scale"),
                ]
                dequantized[wname] = f"{wname}_dequantized"
                dropped.add(wname)
                nodes.append(
                    onnx.helper.make_node(
                        "DequantizeLinear",
                        [f"{wname}_int8", f"{wname}_scale"],
                        [dequantized[wname]],
                        axis=axis,
                    )
                )
            inputs = list(node.input)
            inputs[weight_input] = dequantized[wname]
            nodes.append(onnx.helper.make_node("MatMul", inputs, list(node.output), name=node.name))
        else:
            nodes.append(node)
    model.graph.ClearField("node")
    model.graph.node.extend(nodes)
    kept = {name for node in model.graph.node for name in node.input if name in initializers}
    model.graph.ClearField("initializer")
    model.graph.initializer.extend(
        tensor for name, tensor in initializers.items() if name not in dropped or name in kept
    )
    model.graph.initializer.extend(added)
    rewritten = graph.with_suffix(".rewritten.onnx")
    onnx.save_model(model, str(rewritten))
    rewritten.replace(graph)


def quantize_dynamic_int8(graph: Path) -> None:
    """Rewrite *graph* in place with every MatMul weight dynamic-quantized to int8."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantized = graph.with_suffix(".quantized.onnx")
    quantize_dynamic(
        graph,
        quantized,
        op_types_to_quantize=["MatMul"],
        per_channel=True,
        # Seven-bit weights: x86 kernels without VNNI multiply u8 x s8 into a
        # saturating 16-bit lane and full-range int8 overflows it.
        reduce_range=True,
        weight_type=QuantType.QInt8,
    )
    quantized.replace(graph)


def write_tokenizer(tokenizer: BaseTokenizer, out_dir: Path) -> None:
    """Write the base model's ``tokenizer.json`` into *out_dir*, the file the loader reads."""
    save_pretrained = getattr(tokenizer, "save_pretrained", None)
    save = getattr(tokenizer, "save", None)
    if save_pretrained is not None:
        save_pretrained(str(out_dir))
    elif save is not None:
        save(str(out_dir / "tokenizer.json"))
    else:
        raise ValueError("the tokenizer has neither save_pretrained nor save")
    if not (out_dir / "tokenizer.json").is_file():
        raise ValueError("the tokenizer did not write tokenizer.json into the export directory")
