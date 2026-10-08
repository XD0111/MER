# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------------
# Name:         main_async_fewshot
# Description: Asynchronous API testing with optional few-shot / ICL prompting
# Date:        2026/04/29
# -------------------------------------------------------------------------------

import os
import time
import json
import asyncio
import aiohttp
import tqdm
from datetime import datetime
import random
import logging
import platform
from collections import defaultdict

# 解决 Windows 环境下 asyncio 可能会报错的问题
if platform.system() == 'Windows':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from load_data import load_json
from processe_data import trans_llm, get_mention_schema
from parameter import parse_args
from utils import makedir

# ================= 1. 环境与参数设置 =================
# 不建议在代码里明文写 API key。请在命令行中设置环境变量 OPENAI_API_KEY。
# Windows PowerShell:  $env:OPENAI_API_KEY="你的key"
api_key = os.environ.get("OPENAI_API_KEY", "")
os.environ["OPENAI_API_KEY"] = api_key

if not api_key:
    raise ValueError("Please set OPENAI_API_KEY in your environment before running this script.")

# --- 异步并发核心配置 ---
CONCURRENCY_LIMIT = int(os.environ.get("CONCURRENCY_LIMIT", "30"))

# --- few-shot / ICL 配置 ---
# FEW_SHOT_K=0 表示 zero-shot；FEW_SHOT_K=1/3/5 表示对应的 ICL 实验。
FEW_SHOT_K = int(os.environ.get("FEW_SHOT_K", "1"))

# random: 从训练集中随机选 k 个同类型 prompt 作为 demonstrations。
# balanced: 尽量按答案标签均衡选样例；如果不够，再随机补齐。
FEW_SHOT_STRATEGY = os.environ.get("FEW_SHOT_STRATEGY", "random").lower()

args = parse_args()
makedir(args.log)

t = time.strftime('%Y-%m-%d %H_%M_%S', time.localtime())
args.log = args.log + f'__{t}_fewshot{FEW_SHOT_K}_{FEW_SHOT_STRATEGY}_async.txt'

# ================= 2. 日志与随机种子设置 =================
for name in logging.root.manager.loggerDict:
    if 'transformers' in name:
        logging.getLogger(name).setLevel(logging.CRITICAL)

logging.basicConfig(format='%(message)s', level=logging.INFO,
                    filename=args.log,
                    filemode='w')
logger = logging.getLogger(__name__)


def printlog(message: object, printout: object = True) -> object:
    message = '{}: {}'.format(datetime.now(), message)
    if printout:
        print(message)
    logger.info(message)


for attr in vars(args):
    printlog("{}: {}".format(attr, getattr(args, attr)), printout=False)


def setup_seed(seed):
    random.seed(seed)


setup_seed(args.seed)


# ================= 3. Few-shot / ICL 工具函数 =================
def get_candidate_signature(input_text):
    """从 trans_llm 生成的 prompt 中读取 Candidates 行，用于区分当前 prompt 的关系类型。"""
    for line in input_text.splitlines():
        if line.startswith("Candidates :"):
            return line.replace("Candidates :", "").strip()
    return "UNKNOWN"


def build_fewshot_bank(train_llm_data):
    """将训练集 LLM prompt 按 Candidates 类型分组。"""
    bank = defaultdict(list)
    for input_text, label in train_llm_data:
        sig = get_candidate_signature(input_text)
        bank[sig].append((input_text, label))
    return bank


def sample_random(examples, k):
    """随机采样 k 个样例；如果样例不足，则允许重复采样。"""
    if k <= 0:
        return []
    if len(examples) >= k:
        return random.sample(examples, k)
    return [random.choice(examples) for _ in range(k)]


def sample_balanced(examples, k):
    """尽量按答案标签均衡采样。适合 3-shot/5-shot，更不容易全抽到 None。"""
    if k <= 0:
        return []

    by_label = defaultdict(list)
    for item in examples:
        by_label[item[1]].append(item)

    labels = list(by_label.keys())
    random.shuffle(labels)

    selected = []
    # 先每个 label 选一个
    for label in labels:
        if len(selected) >= k:
            break
        selected.append(random.choice(by_label[label]))

    # 如果还不够，再随机补齐
    remaining_pool = [ex for ex in examples if ex not in selected]
    while len(selected) < k:
        if remaining_pool:
            one = random.choice(remaining_pool)
            selected.append(one)
            remaining_pool.remove(one)
        else:
            selected.append(random.choice(examples))

    return selected


def select_fewshot_demos(fewshot_bank, k, strategy="random"):
    """
    为每一种 Candidates 类型预先固定采样 k 个 demonstration。
    这样同一轮实验中，所有测试样本使用同一批 demonstrations，结果更可复现。
    """
    selected = {}
    for sig, examples in fewshot_bank.items():
        if not examples:
            selected[sig] = []
            continue
        if strategy == "balanced":
            selected[sig] = sample_balanced(examples, k)
        else:
            selected[sig] = sample_random(examples, k)
    return selected


def format_fewshot_prompt(target_input, demos):
    """将 demonstrations 拼接到当前测试样本 prompt 前面。"""
    if not demos:
        return target_input

    parts = []
    parts.append(
        "You are given several labeled examples. "
        "Learn the answer format from the examples. "
        "For the final instance, output only one candidate label.\n"
    )

    for idx, (demo_input, demo_label) in enumerate(demos, start=1):
        parts.append(f"Example {idx}:\n{demo_input.strip()}\nAnswer: {demo_label}\n")

    parts.append(f"Now answer the following instance:\n{target_input.strip()}\nAnswer:")
    return "\n".join(parts)


def add_fewshot_to_test_data(test_llm_data, selected_demos):
    """给每条测试 prompt 加上对应 Candidates 类型的 demonstrations。"""
    new_data = []
    for input_text, label in test_llm_data:
        sig = get_candidate_signature(input_text)
        demos = selected_demos.get(sig, [])
        new_input = format_fewshot_prompt(input_text, demos)
        new_data.append([new_input, label])
    return new_data


# ================= 4. 异步请求处理函数 =================
async def fetch_and_save(item, session, semaphore, file_lock, test_file, progress):
    input_text = item[0]
    label = item[1]

    payload = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "user", "content": input_text}
        ]
    }
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }

    max_retries = 3
    retry_delay = 2
    result = "Error"

    url = "https://api.chatanywhere.tech/v1/chat/completions"

    async with semaphore:
        for attempt in range(max_retries):
            try:
                async with session.post(url, json=payload, headers=headers) as res:
                    raw_data = await res.text()

                    if res.status == 200:
                        try:
                            response_json = json.loads(raw_data)
                            if 'choices' in response_json:
                                result = response_json['choices'][0]['message']['content']
                                break
                            else:
                                result = str(response_json)
                                break
                        except json.JSONDecodeError:
                            printlog(f"\n[JSON解析失败] 内容不是JSON: {raw_data[:100]}", printout=False)
                            await asyncio.sleep(retry_delay)

                    elif res.status == 429:
                        printlog(f"\n[HTTP 429 Error] 触发并发限制，等待重试...", printout=False)
                        await asyncio.sleep(retry_delay * 2)

                    else:
                        printlog(f"\n[HTTP {res.status} Error] 尝试 {attempt + 1}/{max_retries} 失败: {raw_data[:100]}",
                                 printout=False)
                        await asyncio.sleep(retry_delay)

            except asyncio.TimeoutError:
                printlog(f"\n[请求超时] 尝试 {attempt + 1}/{max_retries} 失败.", printout=False)
                await asyncio.sleep(retry_delay)
            except Exception as e:
                printlog(f"\n[网络/请求异常] 尝试 {attempt + 1}/{max_retries} 失败: {e}", printout=False)
                await asyncio.sleep(retry_delay)

    if isinstance(result, list):
        try:
            result = "".join(result)
        except TypeError:
            result = json.dumps(result, ensure_ascii=False)
    elif not isinstance(result, str):
        result = str(result)

    final_dec_res = result.replace('\n', '\t')

    async with file_lock:
        test_file.write(f"{final_dec_res}\t{label}\n")
        test_file.flush()
        progress.update(1)


# ================= 5. 异步主函数 =================
async def main():
    printlog(f"Few-shot setting: k={FEW_SHOT_K}, strategy={FEW_SHOT_STRATEGY}")

    printlog("Loading and processing test data...")
    test_data = load_json(args.test_data_path)
    test_data = get_mention_schema(test_data, args)
    test_data = trans_llm(test_data, args)

    if FEW_SHOT_K > 0:
        printlog("Loading and processing train data for few-shot demonstrations...")
        train_data = load_json(args.train_data_path)
        train_data = get_mention_schema(train_data, args)
        train_llm_data = trans_llm(train_data, args)

        fewshot_bank = build_fewshot_bank(train_llm_data)
        selected_demos = select_fewshot_demos(fewshot_bank, FEW_SHOT_K, FEW_SHOT_STRATEGY)

        for sig, demos in selected_demos.items():
            demo_labels = [label for _, label in demos]
            printlog(f"Candidates=[{sig}] selected demo labels: {demo_labels}", printout=False)

        test_data = add_fewshot_to_test_data(test_data, selected_demos)

    test_size = len(test_data)
    printlog(f"Total test samples: {test_size}")

    test_file_path = f'./test_file_fewshot{FEW_SHOT_K}_{FEW_SHOT_STRATEGY}_{t}.txt'

    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    file_lock = asyncio.Lock()

    progress = tqdm.tqdm(total=test_size, ncols=75, desc="Calling API")
    timeout = aiohttp.ClientTimeout(total=30)

    with open(test_file_path, "w", encoding='utf-8') as test_file:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            tasks = [
                fetch_and_save(item, session, semaphore, file_lock, test_file, progress)
                for item in test_data
            ]
            await asyncio.gather(*tasks)

    progress.close()
    printlog(f"Testing finished. Results saved to {test_file_path}")


# ================= 6. 启动点 =================
if __name__ == "__main__":
    asyncio.run(main())
