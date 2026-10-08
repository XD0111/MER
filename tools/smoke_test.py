# -*- coding: utf-8 -*-
"""MER 仓库冒烟测试：不需要 GPU、数据和预训练权重即可运行。

用法：
    python tools/smoke_test.py

检查内容：
  A. 环境与版本（torch / transformers；AdamW 在 transformers>=5 已移除，会给出提示）
  B. 三个入口（mer/、baselines/encoder_baselines/deberta/、baselines/llm_prompt/）
     拉起的完整依赖导入链是否可解析
  C. 四个编码器（RoBERTa/DeBERTa/BERT/ERNIE）在代码中引用的属性路径
     （.roberta/.lm_head、.deberta/.cls、.bert/.cls、.ernie/.cls）是否存在，
     并用微型随机权重做一次真实前向（不从网络下载任何模型）
  D. argparse 参数定义是否合法（--help 可正常输出）
"""
import importlib
import os
import sys
import warnings

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PASS, FAIL = [], []


def report(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" —— {detail}" if detail else ""))


print("=" * 72)
print("A. 环境与版本")
print("=" * 72)
try:
    import torch
    report("torch 可导入", True, torch.__version__)
except Exception as e:  # noqa: BLE001
    report("torch 可导入", False, str(e))
try:
    import transformers
    report("transformers 可导入", True, transformers.__version__)
    try:
        from transformers import AdamW  # noqa: F401
        report("transformers.AdamW 可用", True)
    except ImportError:
        report("transformers.AdamW 不可用", False,
               "transformers>=5 已移除 AdamW。mer/ 与 encoder_baselines 的 main.py "
               "需要 transformers 4.x（建议 >=4.20,<5）；pip install \"transformers<5\"")
except Exception as e:  # noqa: BLE001
    report("transformers 可导入", False, str(e))

print()
print("=" * 72)
print("B. 依赖导入链（含 AdamW 垫片，仅测模块可解析性，不加载权重）")
print("=" * 72)
try:
    from transformers import torch as _  # noqa: F401  确认 torch 后端
except Exception:
    pass

# AdamW 垫片：transformers 5.x 下补一个等价对象，让其余导入链可被完整测试
import transformers as _tf

if not hasattr(_tf, "AdamW"):
    try:
        from torch.optim import AdamW as _TorchAdamW

        _tf.AdamW = _TorchAdamW
        print("  [info] 已注入 AdamW 垫片（torch.optim.AdamW）以继续测试导入链")
    except Exception:  # noqa: BLE001
        pass

CHAIN = {
    "mer/（RoBERTa 主基线）": (["main 的依赖"], os.path.join(ROOT, "mer"),
        ["Models.model", "Models.Router", "Models.my_modules.CGE",
         "Models.Common_Experts.CommonExperts", "Models.Common_Experts.SentenceExpert",
         "Models.Domain_Experts.ContextExpert", "Models.Domain_Experts.PathExpert",
         "Models.Domain_Experts.GraphExpert", "data.data_set", "data.load_data",
         "data.processe_data", "utils", "cal"]),
    "encoder_baselines/deberta/（DeBERTa 基线）": (["eval_main 的依赖"],
        os.path.join(ROOT, "baselines", "encoder_baselines", "deberta"),
        ["Models.model", "Models.eval_model", "Models.Router", "Models.my_modules.CGE",
         "Models.Common_Experts.CommonExperts", "Models.Common_Experts.SentenceExpert",
         "Models.Domain_Experts.ContextExpert", "Models.Domain_Experts.PathExpert",
         "Models.Domain_Experts.GraphExpert", "data.data_set", "data.load_data",
         "data.processe_data", "utils", "cal"]),
    "encoder_baselines/bert/（BERT 基线）": (["main 的依赖"],
        os.path.join(ROOT, "baselines", "encoder_baselines", "bert"),
        ["Models.model", "Models.Router", "Models.my_modules.CGE",
         "Models.Common_Experts.CommonExperts", "Models.Common_Experts.SentenceExpert",
         "Models.Domain_Experts.ContextExpert", "Models.Domain_Experts.PathExpert",
         "Models.Domain_Experts.GraphExpert", "data.data_set", "data.load_data",
         "data.processe_data", "utils", "cal"]),
    "encoder_baselines/ernie/（ERNIE 基线）": (["main 的依赖"],
        os.path.join(ROOT, "baselines", "encoder_baselines", "ernie"),
        ["Models.model", "Models.Router", "Models.my_modules.CGE",
         "Models.Common_Experts.CommonExperts", "Models.Common_Experts.SentenceExpert",
         "Models.Domain_Experts.ContextExpert", "Models.Domain_Experts.PathExpert",
         "Models.Domain_Experts.GraphExpert", "data.data_set", "data.load_data",
         "data.processe_data", "utils", "cal"]),
    "baselines/llm_prompt/（LLM API 基线）": (["main_fewshot 的依赖"],
        os.path.join(ROOT, "baselines", "llm_prompt"),
        ["load_data", "processe_data", "parameter", "utils", "data_set"]),
}

for label, (_, cwd, mods) in CHAIN.items():
    sys.path.insert(0, cwd)
    ok_all, err = True, ""
    for m in mods:
        try:
            importlib.import_module(m)
        except Exception as e:  # noqa: BLE001
            ok_all, err = False, f"{m}: {type(e).__name__}: {e}"
            break
    report(f"{label} 依赖链导入", ok_all, err if not ok_all else f"{len(mods)} 个模块全部可解析")
    sys.path.pop(0)
    for m in list(sys.modules):
        if m.split(".")[0] in ("Models", "data", "load_data", "processe_data",
                               "parameter", "utils", "cal", "data_set"):
            del sys.modules[m]

print()
print("=" * 72)
print("C. 四个编码器的属性路径与微型前向（随机小权重，不联网）")
print("=" * 72)
try:
    import torch
    from transformers import (BertConfig, BertForMaskedLM, DebertaConfig,
                              DebertaForMaskedLM, ErnieConfig, ErnieForMaskedLM,
                              RobertaConfig, RobertaForMaskedLM)

    def tiny(cls, cfg):
        cfg.vocab_size = 120
        cfg.hidden_size = 32
        cfg.num_attention_heads = 2
        cfg.num_hidden_layers = 2
        cfg.intermediate_size = 64
        if hasattr(cfg, "max_position_embeddings"):
            cfg.max_position_embeddings = 64
        if hasattr(cfg, "type_vocab_size"):
            cfg.type_vocab_size = 4
        return cls(cfg)

    def probe(name, cls, cfg, body_attr, head_attr):
        try:
            model = tiny(cls, cfg)
            body = getattr(model, body_attr, None)
            head = getattr(model, head_attr, None)
            assert body is not None, f"缺少主体属性 .{body_attr}"
            assert head is not None, f"缺少输出头 .{head_attr}"
            _ = body.embeddings.word_embeddings.weight
            ids = torch.randint(0, 120, (2, 8))
            hidden = body(ids)[0]
            pro = head(hidden[:, 0, :])                       # (2, V)
            ans = torch.softmax(pro[:, [5, 6, 7]], dim=1)     # answer_space 索引
            assert ans.shape == (2, 3)
            report(name, True, f".{body_attr} / .{head_attr} / embeddings / 前向 OK")
        except Exception as e:  # noqa: BLE001
            report(name, False, str(e))

    probe("RoBERTa（主基线）", RobertaForMaskedLM, RobertaConfig(), "roberta", "lm_head")
    probe("DeBERTa", DebertaForMaskedLM, DebertaConfig(), "deberta", "cls")
    probe("BERT", BertForMaskedLM, BertConfig(), "bert", "cls")
    probe("ERNIE", ErnieForMaskedLM, ErnieConfig(), "ernie", "cls")
except Exception as e:  # noqa: BLE001
    report("编码器属性测试", False, str(e))

print()
print("=" * 72)
print("D. argparse 参数定义（--help 不报错）")
print("=" * 72)
import subprocess  # noqa: E402

for label, cwd in [("mer/parameter.py", os.path.join(ROOT, "mer")),
                   ("encoder_baselines/deberta/parameter.py",
                    os.path.join(ROOT, "baselines", "encoder_baselines", "deberta")),
                   ("baselines/llm_prompt/parameter.py",
                    os.path.join(ROOT, "baselines", "llm_prompt"))]:
    r = subprocess.run([sys.executable, "-c",
                        "import sys; sys.argv=['t','--help'];"
                        "from parameter import parse_args; parse_args()"],
                       cwd=cwd, capture_output=True, text=True)
    report(label, r.returncode == 0,
           (r.stderr.strip().splitlines() or [""])[-1] if r.returncode else "参数定义合法")

print()
print("=" * 72)
total, bad = len(PASS) + len(FAIL), len(FAIL)
print(f"结果：{len(PASS)}/{total} 通过" + (f"，{bad} 项失败" if bad else "，全部通过"))
if FAIL:
    print("失败项：", "; ".join(FAIL))
print("说明：本脚本验证代码结构与依赖完整性；完整训练验证仍需在带数据的机器上运行，")
print("      例如：cd mer && bash run.sh（首次运行会构建 data_cache.pth）")
sys.exit(1 if bad else 0)
