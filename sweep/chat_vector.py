"""Chat vector: adds what instruction tuning changed in Qwen3.5 (chat - base) to a continued-pretraining checkpoint of
the same base, tensor by tensor, and writes the result in the official Qwen3_5ForConditionalGeneration layout, so the
sweep and the explorer load it the way they load Qwen/Qwen3.5-9B.

    out = cpt + scale * (chat - base)

The arithmetic runs in float32 (in bf16 most of the small chat - base difference would round away) and each result is
cast to the chat checkpoint's dtype for that tensor (bf16, except linear_attn.A_log and linear_attn.norm.weight in F32).

A CPT checkpoint saved from Qwen3_5ForCausalLM is text-only: model.layers.*, model.embed_tokens, model.norm, lm_head.
The official checkpoints keep the language model under model.language_model.* and add the vision tower (model.visual.*)
and the multi-token-prediction head (mtp.*). Tensors the CPT checkpoint lacks are copied from chat, which is what
cpt + (chat - base) gives for weights CPT never trained. config.json, the tokenizer (eos <|im_end|>; the base
tokenizer's is <|endoftext|>) and the chat template come from chat as well.

    uv run python sweep/chat_vector.py --cpt /path/to/milestone-5of5 --out models/q35-9b-ms5-cv

--keep leaves tensors at their CPT values, by name substring (--keep embed_tokens,lm_head). The output folder gets
chat_vector.json: the inputs, and per tensor the sizes of the CPT delta (cpt - base) and the chat vector relative to
the base weight and the cosine between the two deltas; a per-layer summary of those is printed.
"""
import argparse
import contextlib
import json
import math
import os
import re
import shutil

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file


def resolve(path, patterns):
    return path if os.path.isdir(path) else snapshot_download(path, allow_patterns=patterns)


def weight_map(root):
    index = os.path.join(root, "model.safetensors.index.json")
    if os.path.exists(index):
        return json.load(open(index))["weight_map"]
    out = {}
    for f in sorted(os.listdir(root)):
        if f.endswith(".safetensors"):
            with safe_open(os.path.join(root, f), "pt") as s:
                out.update({k: f for k in s.keys()})
    return out


def official(name):
    """Text-only checkpoint names -> official names; official names pass through."""
    if name.startswith("model.") and not name.startswith(("model.language_model.", "model.visual.")):
        return "model.language_model." + name[len("model."):]
    return name


def group(name):
    m = re.search(r"layers\.(\d+)\.", name)
    if name.startswith("mtp."):
        return "mtp"
    return f"L{int(m.group(1)):02d}" if m else name.split(".")[-2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpt", required=True, help="continued-pretraining checkpoint of the base (folder)")
    ap.add_argument("--base", default="Qwen/Qwen3.5-9B-Base")
    ap.add_argument("--chat", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--scale", type=float, default=1.0, help="multiplier on the chat vector")
    ap.add_argument("--keep", default="", help="comma-separated name substrings left at their CPT values")
    ap.add_argument("--shard-gb", type=float, default=5.0)
    ap.add_argument("--chunk", type=int, default=1 << 25, help="elements per float32 block of the arithmetic")
    a = ap.parse_args()
    keep = [k for k in a.keep.split(",") if k]

    cpt = resolve(a.cpt, ["*.safetensors", "*.json"])
    base = resolve(a.base, ["*.safetensors", "*.json"])
    chat = resolve(a.chat, ["*.safetensors", "*.json", "*.jinja", "*.txt"])
    maps = {"cpt": weight_map(cpt), "base": weight_map(base), "chat": weight_map(chat)}
    roots = {"cpt": cpt, "base": base, "chat": chat}
    from_cpt = {official(k): k for k in maps["cpt"]}

    with contextlib.ExitStack() as stack:
        handles = {}

        def tensor(which, name):
            path = os.path.join(roots[which], maps[which][name])
            if path not in handles:
                handles[path] = stack.enter_context(safe_open(path, "pt"))
            return handles[path].get_slice(name)

        problems = [f"{k}: not in base" for k in maps["chat"] if k not in maps["base"]]
        problems += [f"{k}: not in chat" for k in maps["base"] if k not in maps["chat"]]
        problems += [f"{k} ({official(k)}): not in chat" for k in maps["cpt"] if official(k) not in maps["chat"]]
        for k in maps["chat"]:
            shapes = [tensor("chat", k).get_shape()]
            if k in maps["base"]:
                shapes.append(tensor("base", k).get_shape())
            if k in from_cpt:
                shapes.append(tensor("cpt", from_cpt[k]).get_shape())
            if any(s != shapes[0] for s in shapes):
                problems.append(f"{k}: shapes {shapes} (chat, base, cpt)")
        if problems:
            raise SystemExit("checkpoints don't line up:\n  " + "\n  ".join(problems[:40]))

        os.makedirs(a.out, exist_ok=True)
        shard, shard_bytes, parts, wmap, total = {}, 0, [], {}, 0
        report, n_copied = {}, 0

        def flush():
            nonlocal shard, shard_bytes
            if shard:
                part = os.path.join(a.out, f"part-{len(parts):05d}.safetensors")
                save_file(shard, part, metadata={"format": "pt"})
                parts.append((part, list(shard)))
                shard, shard_bytes = {}, 0

        for k in maps["chat"]:
            h = tensor("chat", k)
            if k not in from_cpt:  # vision tower, mtp head: CPT never trained them
                out = h[:]
                n_copied += 1
            else:
                c, b = tensor("cpt", from_cpt[k]), tensor("base", k)
                shape = h.get_shape()
                scale = 0.0 if any(s in k for s in keep) else a.scale
                rows = max(1, a.chunk // max(1, math.prod(shape[1:])))
                out, acc = None, [0.0, 0.0, 0.0, 0.0]  # |base|^2, |cpt - base|^2, |chat - base|^2, dot
                for s in range(0, shape[0], rows):
                    hc = h[s:s + rows]
                    cc, bc = c[s:s + rows].float(), b[s:s + rows].float()
                    dc, dh = cc - bc, hc.float() - bc
                    if out is None:
                        out = torch.empty(shape, dtype=hc.dtype)
                    out[s:s + rows] = (cc + scale * dh).to(hc.dtype)
                    for i, x in enumerate(((bc * bc).sum(), (dc * dc).sum(), (dh * dh).sum(), (dc * dh).sum())):
                        acc[i] += float(x)
                nb = math.sqrt(acc[0]) or 1.0
                report[k] = {"cpt_rel": math.sqrt(acc[1]) / nb, "chat_rel": math.sqrt(acc[2]) / nb,
                             "cos": acc[3] / math.sqrt(acc[1] * acc[2]) if acc[1] and acc[2] else 0.0,
                             "kept": scale == 0.0 and a.scale != 0.0}
            size = out.numel() * out.element_size()
            if shard and shard_bytes + size > a.shard_gb * 1e9:
                flush()
            shard[k] = out.contiguous()
            shard_bytes += size
            total += size
        flush()

    for i, (part, names) in enumerate(parts):
        final = f"model-{i + 1:05d}-of-{len(parts):05d}.safetensors"
        os.replace(part, os.path.join(a.out, final))
        wmap.update({n: final for n in names})
    json.dump({"metadata": {"total_size": total}, "weight_map": wmap},
              open(os.path.join(a.out, "model.safetensors.index.json"), "w"), indent=2)
    for f in os.listdir(chat):
        if not f.endswith(".safetensors") and f not in ("model.safetensors.index.json", "README.md", ".gitattributes"):
            src = os.path.join(chat, f)
            if os.path.isfile(src):
                shutil.copyfile(src, os.path.join(a.out, f))
    json.dump({"cpt": os.path.abspath(cpt), "base": a.base, "chat": a.chat, "scale": a.scale, "keep": keep,
               "merged": len(report), "copied_from_chat": n_copied, "tensors": report},
              open(os.path.join(a.out, "chat_vector.json"), "w"), indent=1)

    groups = {}
    for k, r in report.items():
        groups.setdefault(group(k), []).append(r)
    print(f"{'group':<14}{'cpt/base':>10}{'chat/base':>11}{'cos':>8}   (medians over the group's tensors)")
    for g in sorted(groups):
        med = lambda key: sorted(r[key] for r in groups[g])[len(groups[g]) // 2]
        print(f"{g:<14}{med('cpt_rel'):>10.4f}{med('chat_rel'):>11.4f}{med('cos'):>8.3f}")
    print(f"{len(report)} tensors merged at scale {a.scale}, {n_copied} copied from chat, "
          f"{len(parts)} shards ({total / 1e9:.1f} GB) -> {a.out}")


if __name__ == "__main__":
    main()
