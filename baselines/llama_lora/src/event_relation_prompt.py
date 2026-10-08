from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

RELATION_LABELS = ["Sub-event", "Temporal", "Causal", "Coreference"]
RELATION_CANDIDATES = [
    ["Belong to", "Include", "None"],
    ["Before", "After", "None"],
    ["Cause", "Caused by", "None"],
    ["Cof", "None"],
]

DESCRIPTION = (
    "We annotated the events and their relationships in the given document, "
    "and constructed an event graph with the events as nodes and the relationships "
    "between them as edges. There are four types of relations: Sub-event, Temporal, "
    "Causal, Coreference."
)


def load_json_or_jsonl(path: str | Path) -> List[Any]:
    """Load either a JSON list file or a JSONL file."""
    path = Path(path)
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text[0] in "[{":
        try:
            obj = json.loads(text)
            if isinstance(obj, list):
                return obj
            return [obj]
        except json.JSONDecodeError:
            pass
    rows: List[Any] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _node(sample: Any) -> Dict[str, Any]:
    return sample[0]["node"]


def _relation(sample: Any, rel_name: str) -> List[Sequence[str]]:
    return sample[0].get("relation", {}).get(rel_name, []) or []


def _mention(sample: Any, event_id: str) -> str:
    return str(_node(sample)[event_id].get("mention", event_id))


def _sent_id(sample: Any, event_id: str) -> int:
    return int(_node(sample)[event_id].get("sent_id", 0))


def _sentence_to_text(sentence: Any) -> str:
    if isinstance(sentence, list):
        return " ".join(str(x) for x in sentence)
    return str(sentence)


def ensure_sentences(sample: Any) -> None:
    """Ensure sample[0]['sentences'] exists and can be indexed by int or str.

    The original code sometimes relies on graph['sentences']; other times it
    reconstructs sentences from each event node. This function accepts both.
    """
    graph = sample[0]
    if "sentences" in graph and graph["sentences"]:
        return
    sentences: Dict[str, Any] = {}
    for event_id, node in graph.get("node", {}).items():
        sent_id = str(node.get("sent_id", 0))
        if "sentence" in node:
            sentences[sent_id] = node["sentence"]
    graph["sentences"] = sentences


def get_sentence_by_id(sample: Any, sent_id: int) -> str:
    ensure_sentences(sample)
    sentences = sample[0].get("sentences", {})
    value: Any = None
    if isinstance(sentences, dict):
        value = sentences.get(sent_id, sentences.get(str(sent_id), ""))
    elif isinstance(sentences, list):
        value = sentences[sent_id] if 0 <= sent_id < len(sentences) else ""
    else:
        value = sentences
    return _sentence_to_text(value)


def get_mention_schema(data: List[Any], seed: int = 42) -> List[Any]:
    """Build mention-level structure strings from event graph relations.

    This is a cleaned standalone version of the logic in the user's
    processe_data.py: relation directions are randomly flipped so that inverse
    labels such as Include/Belong to, Before/After and Cause/Caused by can appear.
    """
    rng = random.Random(seed)
    processed = copy.deepcopy(data)

    for sample in processed:
        sub_r = [[i[0], i[1], "Include"] if rng.random() > 0.5 else [i[1], i[0], "Belong to"] for i in _relation(sample, "SUB_EVENT")]
        temp_r = [[i[0], i[1], "Before"] if rng.random() > 0.5 else [i[1], i[0], "After"] for i in _relation(sample, "TEMPORAL")]
        cau_r = [[i[0], i[1], "Cause"] if rng.random() > 0.5 else [i[1], i[0], "Caused by"] for i in _relation(sample, "CAUSAL")]
        cof_r = [[i[0], i[1], "Cof"] for i in _relation(sample, "COF_EVENT")]

        all_relations = sub_r + temp_r + cau_r + cof_r
        if len(sample) < 2 or len(sample[1]) < 2:
            sample[0]["mention_schema"] = ""
            continue

        e1, e2 = sample[1][0], sample[1][1]
        frontier = [e1, e2]
        mention_schema: List[List[str]] = []

        while all_relations:
            next_frontier: List[str] = []
            remaining: List[List[str]] = []
            for rel in all_relations:
                h, t, r = rel
                if h in frontier and t not in frontier:
                    next_frontier.append(t)
                    mention_schema.append(rel)
                elif t in frontier and h not in frontier:
                    next_frontier.append(h)
                    mention_schema.append(rel)
                elif h in frontier and t in frontier:
                    mention_schema.append(rel)
                else:
                    remaining.append(rel)
            all_relations = remaining
            frontier = next_frontier
            if not frontier:
                break

        schema_texts = []
        for h, t, r in mention_schema:
            try:
                schema_texts.append(f"{_mention(sample, h)} {r} {_mention(sample, t)}")
            except KeyError:
                continue
        schema_texts.reverse()
        sample[0]["mention_schema"] = " </s> ".join(schema_texts) + (" </s> " if schema_texts else "")
    return processed


def _question(relation_type: str, candidates: List[str], e1: str, e2: str, single: bool = True) -> str:
    suffix = " Please respond with a single word or phrase." if single else ""
    if relation_type == "Sub-event":
        return (
            f"Please determine if there is a Sub-event relationship between Event1 {e1} and Event2 {e2}. "
            f"If there is, respond with \" {candidates[0]} \" or \" {candidates[1]} \" "
            f"( \" {candidates[0]} \" means Event1 is a sub-event of Event2, while \" {candidates[1]} \" means Event2 is a sub-event of Event1 ). "
            f"If there is no such relationship, respond with \" None \".{suffix}"
        )
    if relation_type == "Temporal":
        return (
            f"Please determine if there is a Temporal relationship between Event1 {e1} and Event2 {e2}. "
            f"If there is, respond with \" {candidates[0]} \" or \" {candidates[1]} \" "
            f"( \" {candidates[0]} \" means Event1 occurs before Event2 in time, while \" {candidates[1]} \" means Event2 occurs before Event1 in time ). "
            f"If there is no such relationship, respond with \" None \".{suffix}"
        )
    if relation_type == "Causal":
        return (
            f"Please determine if there is a Causal relationship between Event1 {e1} and Event2 {e2}. "
            f"If there is, respond with \" {candidates[0]} \" or \" {candidates[1]} \" "
            f"( \" {candidates[0]} \" means Event1 is the cause of Event2, while \" {candidates[1]} \" means Event2 is the cause of Event1 ). "
            f"If there is no such relationship, respond with \" None \".{suffix}"
        )
    return (
        f"Please determine if there is a Coreference relationship between Event1 {e1} and Event2 {e2}. "
        f"If there is, respond with \" {candidates[0]} \" ( \" {candidates[0]} \" means Event1 and Event2 refer to the same event ). "
        f"If there is no such relationship, respond with \" None \".{suffix}"
    )


def build_event_prompt(sample: Any, relation_index: int, single: bool = True) -> str:
    relation_type = RELATION_LABELS[relation_index]
    candidates = RELATION_CANDIDATES[relation_index]
    ensure_sentences(sample)

    node = _node(sample)
    if len(sample) < 2 or len(sample[1]) < 2:
        raise ValueError("Each event sample must contain an event pair at sample[1].")
    e1_id, e2_id = sample[1][0], sample[1][1]
    e1, e2 = _mention(sample, e1_id), _mention(sample, e2_id)

    max_sent = max((_sent_id(sample, event_id) for event_id in node.keys()), default=-1)
    sentence_blocks: List[str] = []
    for sent_id in range(max_sent + 1):
        events = [_mention(sample, event_id) for event_id in node.keys() if _sent_id(sample, event_id) == sent_id]
        if events:
            sentence_blocks.append(" , ".join(events) + " : " + get_sentence_by_id(sample, sent_id))

    fields = {
        "Description": DESCRIPTION,
        "Candidates": ", ".join(candidates),
        "Sentences": " ; ".join(sentence_blocks),
        "Structure": str(sample[0].get("mention_schema", "")),
        "Question": _question(relation_type, candidates, e1, e2, single=single),
    }
    prompt = "\n".join(f"{k}: {v}" for k, v in fields.items())
    return prompt.rstrip() + "\nAnswer: "


def relation_answer(sample: Any, relation_index: int) -> str:
    label_vector = sample[2]
    label_id = int(label_vector[relation_index])
    candidates = RELATION_CANDIDATES[relation_index]
    if label_id < 0 or label_id >= len(candidates):
        raise ValueError(f"Bad label id {label_id} for relation {RELATION_LABELS[relation_index]}")
    return candidates[label_id]


def trans_llm(
    data: List[Any],
    eos_token: str = "</s>",
    single: bool = True,
    include_meta: bool = True,
) -> List[Dict[str, Any]]:
    """Convert raw event-graph samples to TRL prompt/completion rows.

    Each original event-pair sample becomes four instruction samples, one for
    Sub-event, Temporal, Causal and Coreference respectively, matching the
    candidate-label design used in the uploaded few-shot/API code.
    """
    rows: List[Dict[str, Any]] = []
    for sample_id, sample in enumerate(data):
        for rel_idx, rel_name in enumerate(RELATION_LABELS):
            answer = relation_answer(sample, rel_idx)
            completion = answer + (eos_token if eos_token and not answer.endswith(eos_token) else "")
            row: Dict[str, Any] = {
                "prompt": build_event_prompt(sample, rel_idx, single=single),
                "completion": completion,
            }
            if include_meta:
                row.update(
                    {
                        "sample_id": sample_id,
                        "relation_type": rel_name,
                        "candidates": RELATION_CANDIDATES[rel_idx],
                        "gold_label": answer,
                    }
                )
            rows.append(row)
    return rows


def add_few_shot_examples(
    target_rows: List[Dict[str, Any]],
    shot_rows: List[Dict[str, Any]],
    k: int,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Prepend k demonstrations to prompts. Mainly for few-shot baselines."""
    if k <= 0:
        return target_rows
    if not shot_rows:
        raise ValueError("few-shot k > 0 but shot_rows is empty.")
    rng = random.Random(seed)
    output: List[Dict[str, Any]] = []
    for idx, row in enumerate(target_rows):
        pool = shot_rows
        selected = rng.sample(pool, k=min(k, len(pool)))
        demos = []
        for j, ex in enumerate(selected, start=1):
            label = str(ex.get("gold_label") or str(ex.get("completion", "")).replace("</s>", "").strip())
            demos.append(f"Example {j}:\n{ex['prompt']}{label}\n")
        new_row = dict(row)
        new_row["prompt"] = "Here are labeled examples for the task.\n\n" + "\n".join(demos) + "\nNow answer the following instance.\n\n" + row["prompt"]
        output.append(new_row)
    return output


def parse_candidate_label(text: str, candidates: Sequence[str]) -> str:
    """Map a free-form generation to one of the allowed candidate labels."""
    clean = (text or "").replace("</s>", " ").replace("<s>", " ").strip()
    clean_lower = " ".join(clean.lower().replace("\n", " ").replace("\t", " ").split())
    ordered = sorted(candidates, key=len, reverse=True)
    for cand in ordered:
        cand_lower = cand.lower()
        if clean_lower == cand_lower or clean_lower.startswith(cand_lower):
            return cand
    for cand in ordered:
        if cand.lower() in clean_lower:
            return cand
    return clean.split()[0] if clean.split() else ""
