# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------------
# Name:         main
# Description:  Cleaned up for API testing only
# Author:       梁超
# Date:         2026/03/06 (Updated with Retry & Status Validation)
# -------------------------------------------------------------------------------

import os
import time
import http.client
import json
import tqdm
from datetime import datetime
import random
import logging

# 仅保留数据处理相关的自有模块导入
from load_data import load_json
from processe_data import trans_llm, get_mention_schema
from parameter import parse_args
from utils import makedir

# ================= 1. 环境与参数设置 =================
# 直接定义 API Key 变量，方便后续 headers 引用
api_key = os.environ.get("OPENAI_API_KEY", "")
os.environ["OPENAI_API_KEY"] = api_key

args = parse_args()  # load parameters
makedir(args.log)

t = time.strftime('%Y-%m-%d %H_%M_%S', time.localtime())
args.log = args.log + '__' + t + '.txt'

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
    printlog("{}: {}".format(attr, getattr(args, attr)))


def setup_seed(seed):
    random.seed(seed)


setup_seed(args.seed)

# ================= 3. 数据加载与处理 =================
printlog("Loading and processing test data...")
test_data = load_json(args.test_data_path)
test_data = get_mention_schema(test_data, args)
test_data = trans_llm(test_data, args)

test_size = len(test_data)
printlog(f"Total test samples: {test_size}")

# ================= 4. API 请求与结果保存 =================
test_file_path = f'./test_file_{t}.txt'
test_file = open(test_file_path, "w", encoding='utf-8')

progress = tqdm.tqdm(total=test_size, ncols=75, desc="Calling API")

for item in test_data:
    input_text = item[0]
    label = item[1]

    # 构建标准 OpenAI 格式的 payload
    payload = json.dumps({
        "model": "deepseek-v3.2",  # DeepSeek V3 在大部分中转平台的标准名称
        "messages": [
            {"role": "user", "content": input_text}
        ]
    })
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }

    max_retries = 3
    retry_delay = 2
    result = "Error"

    for attempt in range(max_retries):
        try:
            # 增加 timeout 防止请求一直挂起
            conn = http.client.HTTPSConnection("api.chatanywhere.tech", timeout=20)
            # 请求路径修改为标准路径 /v1/chat/completions
            conn.request("POST", "/v1/chat/completions", payload, headers)
            res = conn.getresponse()

            # 读取原始返回数据
            raw_data = res.read().decode("utf-8")

            # 核心防御：必须先判断状态码是不是 200 (成功)，拦截 567、502 等网关错误
            if res.status == 200:
                try:
                    response_json = json.loads(raw_data)
                    if 'choices' in response_json:
                        result = response_json['choices'][0]['message']['content']
                        break  # 成功获取，跳出重试循环
                    else:
                        result = str(response_json)
                        break
                except json.JSONDecodeError:
                    printlog(f"\n[JSON解析失败] 返回状态200但内容不是JSON: {raw_data[:200]}")
                    time.sleep(retry_delay)
            else:
                # 遇到非 200 状态码，直接打印出平台返回的具体错误文字，且不执行 json.loads
                printlog(
                    f"\n[HTTP {res.status} Error] 尝试 {attempt + 1}/{max_retries} 失败. 返回内容: {raw_data[:200]}")
                time.sleep(retry_delay)

        except Exception as e:
            printlog(f"\n[网络/请求异常] 尝试 {attempt + 1}/{max_retries} 失败: {e}")
            time.sleep(retry_delay)
        finally:
            if 'conn' in locals():
                conn.close()

    # 处理多维列表或异常情况的数据类型
    if isinstance(result, list):
        try:
            result = "".join(result)
        except TypeError:
            result = json.dumps(result, ensure_ascii=False)
    elif not isinstance(result, str):
        result = str(result)

    # 替换所有的换行符为制表符，避免破坏你的 txt 文件结构
    final_dec_res = result.replace('\n', '\t')

    # 写入结果并强制落盘，防止程序意外崩溃丢失数据
    test_file.write(f"{final_dec_res}\t{label}\n")
    test_file.flush()

    progress.update(1)

test_file.close()
progress.close()
printlog(f"Testing finished. Results saved to {test_file_path}")