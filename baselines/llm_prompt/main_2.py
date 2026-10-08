# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------------
# Name:         main_async
# Description:  Asynchronous version for fast API testing
# Author:       梁超
# Date:         2026/03/06 (Async + Retry & Status Validation)
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

# 解决 Windows 环境下 asyncio 可能会报错的问题
if platform.system() == 'Windows':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# 仅保留数据处理相关的自有模块导入
from load_data import load_json
from processe_data import trans_llm, get_mention_schema
from parameter import parse_args
from utils import makedir

# ================= 1. 环境与参数设置 =================
api_key = os.environ.get("OPENAI_API_KEY", "")
os.environ["OPENAI_API_KEY"] = api_key

# --- 异步并发核心配置 ---
# 注意：ChatAnywhere 这类中转站通常有 QPS 限制，不要设置得太大。
# 建议先用 20 跑一下，如果不报错再逐渐上调到 50。
CONCURRENCY_LIMIT = 30

args = parse_args()
makedir(args.log)

t = time.strftime('%Y-%m-%d %H_%M_%S', time.localtime())
args.log = args.log + '__' + t + '_async.txt'

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


# ================= 3. 异步请求处理函数 =================
async def fetch_and_save(item, session, semaphore, file_lock, test_file, progress):
    input_text = item[0]
    label = item[1]

    # aiohttp 支持直接传字典进去作为 json payload
    payload = {
        "model": "deepseek-v3.2",
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

    # 使用信号量控制并发数
    async with semaphore:
        for attempt in range(max_retries):
            try:
                # 发起异步 POST 请求
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
                        # 429 代表并发太高触发了速率限制，增加退避等待时间
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

    # 数据格式清洗
    if isinstance(result, list):
        try:
            result = "".join(result)
        except TypeError:
            result = json.dumps(result, ensure_ascii=False)
    elif not isinstance(result, str):
        result = str(result)

    final_dec_res = result.replace('\n', '\t')

    # 使用锁确保多协程写入文件时不会发生错乱
    async with file_lock:
        test_file.write(f"{final_dec_res}\t{label}\n")
        test_file.flush()
        progress.update(1)


# ================= 4. 异步主函数 =================
async def main():
    printlog("Loading and processing test data...")
    test_data = load_json(args.test_data_path)
    test_data = get_mention_schema(test_data, args)
    test_data = trans_llm(test_data, args)

    test_size = len(test_data)
    printlog(f"Total test samples: {test_size}")

    test_file_path = f'./test_file_{t}.txt'

    # 限制并发数量的信号量
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    # 文件写入锁
    file_lock = asyncio.Lock()

    progress = tqdm.tqdm(total=test_size, ncols=75, desc="Calling API")

    # aiohttp 推荐全局使用一个 ClientSession，并设置整体的超时
    timeout = aiohttp.ClientTimeout(total=30)

    with open(test_file_path, "w", encoding='utf-8') as test_file:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            # 构建所有的任务
            tasks = [
                fetch_and_save(item, session, semaphore, file_lock, test_file, progress)
                for item in test_data
            ]

            # 并发执行所有任务
            await asyncio.gather(*tasks)

    progress.close()
    printlog(f"Testing finished. Results saved to {test_file_path}")


# ================= 5. 启动点 =================
if __name__ == "__main__":
    asyncio.run(main())