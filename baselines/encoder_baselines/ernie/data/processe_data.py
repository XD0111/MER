# -*- coding: utf-8 -*-#
import copy
import tqdm
import torch
import random
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from data.data_set import MyDataset
from data.load_data import load_json
from tokenizers import AddedToken
import os


# -------------------------------------------------------------------------------
# Name:         processe-data
# Description:
# Author:       梁超
# Date:         2024/5/16
# -------------------------------------------------------------------------------

def get_mention_schema(data, tokenizer, args):
    process = tqdm.tqdm(total=len(data), ncols=75, desc='Mention schema')
    for d in range(len(data)):
        mention_schema = []
        process.update(1)
        sub_r = [[i[0], i[1], '<su1>'] if random.random() > 0.5 else [i[1], i[0], '<su2>'] for i in
                 data[d][0]['relation']['SUB_EVENT']]
        temp_r = [[i[0], i[1], '<te1>'] if random.random() > 0.5 else [i[1], i[0], '<te2>'] for i in
                  data[d][0]['relation']['TEMPORAL']]
        cau_r = [[i[0], i[1], '<ca1>'] if random.random() > 0.5 else [i[1], i[0], '<ca2>'] for i in
                 data[d][0]['relation']['CAUSAL']]
        cof_r = [[i[0], i[1], '<co1>'] for i in data[d][0]['relation']['COF_EVENT']]

        all_relations = sub_r + temp_r + cau_r + cof_r

        e1, e2 = data[d][1][0], data[d][1][1]
        temp = [e1, e2]
        while len(all_relations) > 0:
            temp2 = []
            for i in all_relations:
                if i[0] in temp and i[1] not in temp:
                    temp2.append(i[1])
                    mention_schema.append(i)
                    all_relations.remove(i)
                elif i[1] in temp and i[0] not in temp:
                    temp2.append(i[0])
                    mention_schema.append(i)
                    all_relations.remove(i)
                elif i[0] in temp and i[1] in temp:
                    mention_schema.append(i)
                    all_relations.remove(i)
            temp = temp2
            if len(temp) == 0:
                break

        mention_schema = [data[d][0]['node'][j[0]]['mention'] + ' ' + j[2] + ' ' + data[d][0]['node'][j[1]]['mention']
                          for j in mention_schema]
        mention_schema.reverse()
        data[d][0]['mention_schema'] = ' [SEP] '.join(mention_schema) + ' [SEP] ' + data[d][0]['node'][e1][
            'mention'] + ' [MASK] ' + data[d][0]['node'][e2]['mention']
        while len(tokenizer.encode(data[d][0]['mention_schema'])) >= args.len_schema - 12:
            l = len(mention_schema)
            mention_schema = mention_schema[int(0.2 * l):]
            data[d][0]['mention_schema'] = ' [SEP] '.join(mention_schema) + ' [SEP] ' + data[d][0]['node'][e1][
                'mention'] + ' [MASK] ' + data[d][0]['node'][e2]['mention']
    return data


def get_type_schema(data, tokenizer, args):
    origin_len = []
    determine_len = []

    process = tqdm.tqdm(total=len(data), ncols=75, desc='Type schema')
    for d in range(len(data)):
        type_schema = []
        process.update(1)
        sub_r = [[i[0], i[1], '<su1>'] if random.random() > 0.5 else [i[1], i[0], '<su2>'] for i in
                 data[d][0]['relation']['SUB_EVENT']]
        temp_r = [[i[0], i[1], '<te1>'] if random.random() > 0.5 else [i[1], i[0], '<te2>'] for i in
                  data[d][0]['relation']['TEMPORAL']]
        cau_r = [[i[0], i[1], '<ca1>'] if random.random() > 0.5 else [i[1], i[0], '<ca2>'] for i in
                 data[d][0]['relation']['CAUSAL']]
        cof_r = [[i[0], i[1], '<co1>'] for i in data[d][0]['relation']['COF_EVENT']]

        all_relations = sub_r + temp_r + cau_r + cof_r
        for i in range(len(all_relations)):
            all_relations[i] = [data[d][0]['node'][all_relations[i][0]]['type'],
                                data[d][0]['node'][all_relations[i][1]]['type'], all_relations[i][2]]

        if args.diff_type == 1:
            all_relations = [tuple(i) for i in all_relations]
            all_relations = list(set(all_relations))
            all_relations = [list(i) for i in all_relations]

        e1, e2 = data[d][0]['node'][data[d][1][0]]['type'], data[d][0]['node'][data[d][1][1]]['type']
        temp = [e1, e2]
        while len(all_relations) > 0:
            temp2 = []
            for i in all_relations:
                if i[0] in temp and i[1] not in temp:
                    temp2.append(i[1])
                    if i not in type_schema:
                        type_schema.append(i)
                    all_relations.remove(i)
                elif i[1] in temp and i[0] not in temp:
                    temp2.append(i[0])
                    if i not in type_schema:
                        type_schema.append(i)
                    all_relations.remove(i)
                elif i[0] in temp and i[1] in temp:
                    if i not in type_schema:
                        type_schema.append(i)
                    all_relations.remove(i)
            temp = temp2
            if len(temp) == 0:
                break

        type_schema = [j[0] + ' ' + j[2] + ' ' + j[1] for j in type_schema]
        type_schema.reverse()
        data[d][0]['type_schema'] = ' [SEP] '.join(type_schema) + ' [SEP] ' + e1 + ' [MASK] ' + e2
        origin_len.append(len(type_schema))
        while len(tokenizer.encode(data[d][0]['type_schema'])) >= args.len_schema - 12:
            l = len(type_schema)
            type_schema = type_schema[int(0.2 * l):]
            data[d][0]['type_schema'] = ' [SEP] '.join(type_schema) + ' [SEP] ' + e1 + ' [MASK] ' + e2
        determine_len.append(len(type_schema))
    return data


def modify_sentences(data):
    # ... 函数内容保持不变 ...
    process = tqdm.tqdm(total=len(data), ncols=75, desc='Modify')
    for d in range(len(data)):
        process.update(1)
        sentence_lists = {}
        for i in data[d][0]['node'].keys():
            sent_id = data[d][0]['node'][i]['sent_id']
            sentence_lists[str(sent_id)] = data[d][0]['node'][i]['sentence']
        data[d][0]['sentences'] = sentence_lists
    return data


def get_common_prompt(data, tokenizer, args):
    # ... 函数内容保持不变 ...
    process = tqdm.tqdm(total=len(data), ncols=75, desc='Common schema')
    for d in range(len(data)):
        process.update(1)
        e1, e2 = data[d][1][0], data[d][1][1]
        data[d][0]['common_schema'] = 'According to the common knowledge, the relation between ' + \
                                      data[d][0]['node'][e1]['mention'] + ' and ' + data[d][0]['node'][e1][
                                          'mention'] + ' is  [MASK] '
    process.close()
    return data


def get_core_event_prompt(data, tokenizer, args):
    # ... 函数内容保持不变 ...
    process = tqdm.tqdm(total=len(data), ncols=75, desc='Core schema')
    for d in range(len(data)):
        process.update(1)
        e1, e2 = data[d][1][0], data[d][1][1]
        data[d][0]['core_event_schema'] = '[In-context sentences]: ' + ' '.join(
            data[d][0]['node'][e1]['sentence']) + ' '.join(
            data[d][0]['node'][e2]['sentence']) + '[Question]: Please infer the relation between' + \
                                          data[d][0]['node'][e1]['mention'] + 'and' + data[d][0]['node'][e2][
                                              'mention'] + 'based on their relation between them and core title' + \
                                          data[d][0]['title'] + '. The relation between ' + data[d][0]['node'][e1][
                                              'mention'] + ' and ' + data[d][0]['node'][e1]['mention'] + ' is  [MASK] '

    process.close()
    return data

def get_document_prompt(data, tokenizer, args):
    process = tqdm.tqdm(total=len(data), ncols=75, desc='Document schema')
    for d in range(len(data)):
        process.update(1)
        e1, e2 = data[d][1][0], data[d][1][1]

        # 1. 获取 e1 和 e2 的所在句子的 sent_id，并确定截取的起止范围
        sent_id1 = data[d][0]['node'][e1]['sent_id']
        sent_id2 = data[d][0]['node'][e2]['sent_id']

        start_id = min(sent_id1, sent_id2)
        end_id = max(sent_id1, sent_id2)

        # 2. 收集落在 [start_id, end_id] 区间内的所有句子，按 sent_id 去重
        sentences_dict = {}
        for node_id, node_info in data[d][0]['node'].items():
            sent_id = node_info['sent_id']
            # 仅保留处于 e1 和 e2 之间的句子
            if start_id <= sent_id <= end_id:
                if sent_id not in sentences_dict:
                    sentences_dict[sent_id] = node_info['sentence']

        # 3. 将过滤后的句子按 sent_id 升序排列
        sorted_sent_ids = sorted(list(sentences_dict.keys()))
        context_text = ' '.join([' '.join(sentences_dict[sid]) for sid in sorted_sent_ids])

        # ================= 新增：保两头，砍中间 动态截断保护 =================
        # 拆分出绝对不能被截断的头部(Title)和尾部(Question)。
        # 修复了掩码格式，严格使用 <mask>
        head_text = '[Title]: ' + data[d][0]['title'] + ' [Context]: '
        tail_text = ' [Question]: Please infer the relation between ' + data[d][0]['node'][e1]['mention'] + ' and ' + \
                    data[d][0]['node'][e2]['mention'] + ' is [MASK]'

        # 使用 tokenizer 估算它们各自的 token 长度
        head_tokens = tokenizer.tokenize(head_text)
        tail_tokens = tokenizer.tokenize(tail_text)
        context_tokens = tokenizer.tokenize(context_text)

        # 留给 Context (正文) 的最大 token 容量
        # (减去 2 是为了给 RoBERTa 默认添加的 <s> 和 </s> 留位置)
        max_context_len = args.len_schema - len(head_tokens) - len(tail_tokens) - 15

        if max_context_len <= 0:
            # 极端情况：标题和问题加起来已超标，直接把 Context 置空
            context_tokens = []
        elif len(context_tokens) > max_context_len:
            # 发生超长：执行【保两头，砍中间】策略
            dot_tokens = tokenizer.tokenize(' ... ')  # 中间用省略号连接

            # 计算扣除省略号后，前后各能保留多少个 token
            available_len = max_context_len - len(dot_tokens)
            if available_len > 0:
                half_len = available_len // 2
                first_half = context_tokens[:half_len]
                second_half = context_tokens[-(available_len - half_len):]
                # 重新拼接：前一段 + 省略号 + 后一段
                context_tokens = first_half + dot_tokens + second_half
            else:
                context_tokens = context_tokens[:max_context_len]

        # 将处理好的安全 token 列表转回纯文本字符串
        truncated_context = tokenizer.convert_tokens_to_string(context_tokens)
        # ========================================================

        # 4. 构建最终安全且无损关键信息的 Prompt
        data[d][0]['document_schema'] = head_text + truncated_context + tail_text

    process.close()
    return data


def get_key_schema(data, tokenizer, args):
    # ... 函数内容保持不变 ...
    process = tqdm.tqdm(total=len(data), ncols=75, desc='Key schema')
    for d in range(len(data)):
        process.update(1)
        e1, e2 = data[d][1][0], data[d][1][1]
        data[d][0]['key_schema'] = '[Title]: ' + data[d][0]['title'] + '[In-context sentences]: ' + ' '.join(
            data[d][0]['node'][e1]['sentence']) + ' '.join(
            data[d][0]['node'][e2]['sentence']) + '[Question]: Please infer the relation between' + \
                                   data[d][0]['node'][e1]['mention'] + 'and' + data[d][0]['node'][e2]['mention'] + '.'

    process.close()
    return data


def collect_mult_event(data, tokenizer):
    # ... 函数内容保持不变 ...
    process = tqdm.tqdm(total=len(data), ncols=75, desc='Collecting')
    multi_event = []
    to_add = {}
    special_multi_event_token = []
    event_dict = {}
    reverse_event_dict = {}
    for d in data:
        process.update(1)
        for event_id in d[0]['node'].keys():
            mention = d[0]['node'][event_id]['mention']
            if len(tokenizer(' ' + mention)['input_ids'][1:-1]) > 1 and mention not in multi_event:
                multi_event.append(mention)
                special_multi_event_token.append("<a_" + str(len(special_multi_event_token)) + ">")
                event_dict[special_multi_event_token[-1]] = multi_event[-1]
                reverse_event_dict[multi_event[-1]] = special_multi_event_token[-1]
                to_add[special_multi_event_token[-1]] = tokenizer(multi_event[-1].strip())['input_ids'][1: -1]

    process.close()
    return multi_event, special_multi_event_token, event_dict, reverse_event_dict, to_add


def replace_mult_event(data, reverse_event_dict):
    # ... 函数内容保持不变 ...
    process = tqdm.tqdm(total=len(data), ncols=75, desc='Replacing')
    for d in range(len(data)):
        process.update(1)

        event_id_list = list(data[d][0]['node'].keys())
        sorted_event_ids = sorted(event_id_list, key=lambda x: (
                    1000 * data[d][0]['node'][x]['sent_id'] + data[d][0]['node'][x]['location'][0]), reverse=True)

        for event_id in sorted_event_ids:
            sent_id = data[d][0]['node'][event_id]['sent_id']
            mention = data[d][0]['node'][event_id]['mention']
            sentence = data[d][0]['sentences'][str(sent_id)]
            location = data[d][0]['node'][event_id]['location']

            if mention in reverse_event_dict:
                data[d][0]['node'][event_id]['mention'] = reverse_event_dict[mention]
                data[d][0]['sentences'][str(sent_id)] = sentence[:location[0]] + [
                    reverse_event_dict[mention]] + sentence[location[1]:]
                data[d][0]['node'][event_id]['location'] = [location[0], location[0] + 1]

    process.close()
    return data


def insert_event_marks(data, mark='<c>'):
    # ... 函数内容保持不变 ...
    process = tqdm.tqdm(total=len(data), ncols=75, desc='Inserting')
    for d in range(len(data)):
        process.update(1)

        event_id_list = list(data[d][0]['node'].keys())
        sorted_event_ids = sorted(event_id_list, key=lambda x: (
                    1000 * data[d][0]['node'][x]['sent_id'] + data[d][0]['node'][x]['location'][0]), reverse=True)

        pre_sen_id = data[d][0]['node'][sorted_event_ids[0]]['sent_id']
        sent_loc = 0
        for event_id in sorted_event_ids:
            cur_sen_id = data[d][0]['node'][event_id]['sent_id']
            if cur_sen_id != pre_sen_id:
                sent_loc = 0

            mention = data[d][0]['node'][event_id]['mention']
            sentence = data[d][0]['sentences'][str(cur_sen_id)]
            location = data[d][0]['node'][event_id]['location']

            data[d][0]['sentences'][str(cur_sen_id)] = sentence[:location[0]] + [mark, mention, mark] + sentence[
                location[1]:]
            data[d][0]['node'][event_id]['location'] = sent_loc
            sent_loc += 1
            pre_sen_id = cur_sen_id
    return data


def simplify_data(data, args, tokenizer):
    # ... 函数内容保持不变，但需要更新硬编码的token ID ...
    process = tqdm.tqdm(total=len(data), ncols=75, desc='Simplifying')
    for d in range(len(data)):
        process.update(1)
        events_to_ids = data[d][0]['events_to_ids']
        ids_to_events = data[d][0]['ids_to_events']
        idxs = []
        masks = []
        candi_locs = []
        locs = []
        sent_ids = []
        adjacency_matrix = torch.tensor(data[d][0]['adjacency'])

        raw_sentences = data[d][0]['sentences']
        raw_to_new_sent_ids = {}
        new_to_raw_sent_ids = {}

        t = 0
        for i in raw_sentences.keys():
            raw_to_new_sent_ids[i] = t
            new_to_raw_sent_ids[str(t)] = i
            t += 1
            prompt = 'Events modeling: ' + ' '.join(raw_sentences[i])
            encode_dict_sub = tokenizer.encode_plus(
                prompt,
                add_special_tokens=True,
                padding='max_length',
                max_length=args.len_arg,
                truncation=True,
                pad_to_max_length=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            idx = encode_dict_sub['input_ids']
            mask = encode_dict_sub['attention_mask']
            # 更新：使用tokenizer转换token为ID，而不是硬编码
            candi_loc = torch.nonzero(idx == tokenizer.convert_tokens_to_ids('<c>'), as_tuple=False)[:, 1][::2]

            if len(idxs) == 0:
                idxs = idx
                masks = mask
            else:
                idxs = torch.cat((idxs, idx), dim=0)
                masks = torch.cat((masks, mask), dim=0)
            candi_locs.append(candi_loc)

        for i in range(len(ids_to_events)):
            e_ids = ids_to_events[i]
            sent_ids.append(raw_to_new_sent_ids[str(data[d][0]['node'][e_ids]['sent_id'])])
            cands = candi_locs[raw_to_new_sent_ids[str(data[d][0]['node'][e_ids]['sent_id'])]]
            loc = data[d][0]['node'][e_ids]['location']
            l = len(cands) - 1
            locs.append(int(cands[l - loc]) + 1)

        events_schema = tokenizer.encode_plus(
            data[d][0]['mention_schema'],
            add_special_tokens=True,
            padding='max_length',
            max_length=args.len_schema,
            truncation=True,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        idx_event_schema = events_schema['input_ids']
        mask_event_schema = events_schema['attention_mask']
        # 更新：使用tokenizer的mask_token_id
        event_mention_loc = torch.nonzero(idx_event_schema == tokenizer.mask_token_id, as_tuple=False)[0][1]

        type_schema = tokenizer.encode_plus(
            data[d][0]['type_schema'],
            add_special_tokens=True,
            padding='max_length',
            max_length=args.len_schema,
            truncation=True,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        idx_type_schema = type_schema['input_ids']
        mask_type_schema = type_schema['attention_mask']
        event_type_loc = torch.nonzero(idx_type_schema == tokenizer.mask_token_id, as_tuple=False)[0][1]

        common_schema = tokenizer.encode_plus(
            data[d][0]['common_schema'],
            add_special_tokens=True,
            padding='max_length',
            max_length=args.len_common_schema,
            truncation=True,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        idx_common_schema = common_schema['input_ids']
        mask_common_schema = common_schema['attention_mask']
        event_common_loc = torch.nonzero(idx_common_schema == tokenizer.mask_token_id, as_tuple=False)[0][1]

        document_schema = tokenizer.encode_plus(
            data[d][0]['document_schema'],
            add_special_tokens=True,
            padding='max_length',
            max_length=args.len_schema,  # 沿用普通的 len_schema 长度限制
            truncation=True,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        idx_document_schema = document_schema['input_ids']
        mask_document_schema = document_schema['attention_mask']
        event_document_loc = torch.nonzero(idx_document_schema == tokenizer.mask_token_id, as_tuple=False)[0][1]

        core_events_schema = tokenizer.encode_plus(
            data[d][0]['core_event_schema'],
            add_special_tokens=True,
            padding='max_length',
            max_length=args.len_schema,
            truncation=True,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        idx_core_schema = core_events_schema['input_ids']
        mask_core_schema = core_events_schema['attention_mask']
        event_core_loc = torch.nonzero(idx_core_schema == tokenizer.mask_token_id, as_tuple=False)[0][1]

        key_schema = tokenizer.encode_plus(
            data[d][0]['key_schema'],
            add_special_tokens=True,
            padding='max_length',
            max_length=args.len_key_schema,
            truncation=True,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        idx_key_schema = key_schema['input_ids']
        mask_key_schema = key_schema['attention_mask']

        # path1_schema = tokenizer.encode_plus(
        #     data[d][0]['path1'],
        #     add_special_tokens=True,
        #     padding='max_length',
        #     max_length=args.len_path_schema,
        #     truncation=True,
        #     pad_to_max_length=True,
        #     return_attention_mask=True,
        #     return_tensors='pt'
        # )
        # idx_path1_schema = path1_schema['input_ids']
        # mask_path1_schema = path1_schema['attention_mask']
        #
        # path2_schema = tokenizer.encode_plus(
        #     data[d][0]['path2'],
        #     add_special_tokens=True,
        #     padding='max_length',
        #     max_length=args.len_path_schema,
        #     truncation=True,
        #     pad_to_max_length=True,
        #     return_attention_mask=True,
        #     return_tensors='pt'
        # )
        # idx_path2_schema = path2_schema['input_ids']
        # mask_path2_schema = path2_schema['attention_mask']

        n = {'idx': idxs,
             'mask': masks,
             'sentence_ids': sent_ids,
             'location': locs,
             'event_schema': idx_event_schema,
             'event_schema_mask': mask_event_schema,
             'common_schema': idx_common_schema,
             'common_schema_mask': mask_common_schema,
             'key_schema': idx_key_schema,
             'key_schema_mask': mask_key_schema,
             'document_schema': idx_document_schema,
             'document_schema_mask': mask_document_schema,
             'event_document_loc': event_document_loc,
             'core_schema': idx_core_schema,
             'core_schema_mask': mask_core_schema,
             'event_mention_loc': event_mention_loc,
             'event_type_loc': event_type_loc,
             'type_schema': idx_type_schema,
             'type_schema_mask': mask_type_schema,
             'event_common_loc': event_common_loc,
             'event_core_loc': event_core_loc,
             }

        data[d] = [n, adjacency_matrix, [events_to_ids[data[d][1][0]], events_to_ids[data[d][1][1]]], data[d][2]]
    return data


def longest_path_from_node(adj_matrix, start_node, max_depth=10):
    # ... 函数内容保持不变 ...
    num_nodes = len(adj_matrix)
    visited = [False] * num_nodes
    max_len = [0]
    best_path = []

    def dfs(node, path, depth):
        if len(path) > max_len[0]:
            max_len[0] = depth
            best_path.clear()
            best_path.extend(path)

        if len(path) >= max_depth:
            return

        visited[node] = True
        for neighbor in range(num_nodes):
            if node == neighbor:
                continue
            if (adj_matrix[node][neighbor] > 0 or adj_matrix[neighbor][node] > 0) and not visited[
                neighbor] and neighbor not in path:
                path.append(neighbor)
                dfs(neighbor, path, depth + 1)
                if len(path) == 10:
                    max_len[0] = 10
                    best_path.clear()
                    best_path.extend(path)
                    return
                path.pop()
        visited[node] = False

    dfs(start_node, [start_node], 1)

    return max_len[0], best_path


def get_path_schema(data, tokenizer, args):
    # ... 函数内容保持不变 ...
    process = tqdm.tqdm(total=len(data), ncols=75, desc='Pathing')
    for d in range(len(data)):
        process.update(1)
        e1, e2 = data[d][1][0], data[d][1][1]
        e1_id, e2_id = data[d][0]['events_to_ids'][e1], data[d][0]['events_to_ids'][e2]
        adjacency_matrix = copy.deepcopy(data[d][0]['adjacency'][0])
        for i in range(len(data[d][0]['adjacency'][0])):
            for j in range(len(data[d][0]['adjacency'][0][0])):
                adjacency_matrix[i][j] = data[d][0]['adjacency'][0][i][j] + data[d][0]['adjacency'][1][i][j] + \
                                         data[d][0]['adjacency'][2][i][j] + data[d][0]['adjacency'][3][i][j]
        l1, path1 = longest_path_from_node(adjacency_matrix, e1_id)
        l2, path2 = longest_path_from_node(adjacency_matrix, e2_id)
        path1.reverse()
        path2.reverse()

        path1_prompt = [path1[0]]
        for i in path1[1:]:
            pre = path1_prompt[-1]
            if data[d][0]['adjacency'][0][i][pre] > 0:
                path1_prompt.append('<su1>')
                path1_prompt.append(i)
            elif data[d][0]['adjacency'][0][pre][i] > 0:
                path1_prompt.append('<su2>')
                path1_prompt.append(i)
            elif data[d][0]['adjacency'][1][i][pre] > 0:
                path1_prompt.append('<te1>')
                path1_prompt.append(i)
            elif data[d][0]['adjacency'][1][pre][i] > 0:
                path1_prompt.append('<te2>')
                path1_prompt.append(i)
            elif data[d][0]['adjacency'][2][i][pre] > 0:
                path1_prompt.append('<ca1>')
                path1_prompt.append(i)
            elif data[d][0]['adjacency'][2][pre][i] > 0:
                path1_prompt.append('<ca2>')
                path1_prompt.append(i)
            elif data[d][0]['adjacency'][3][i][pre] > 0 or data[d][0]['adjacency'][3][pre][i] > 0:
                path1_prompt.append('<co1>')
                path1_prompt.append(i)
        for i in range(len(path1_prompt)):
            if i % 2 == 0:
                if i == 0:
                    path1_prompt[i] = ' ' + data[d][0]['node'][data[d][0]['ids_to_events'][path1_prompt[i]]]['mention']
                else:
                    path1_prompt[i] = data[d][0]['node'][data[d][0]['ids_to_events'][path1_prompt[i]]]['mention']

        path2_prompt = [path2[0]]
        for i in path2[1:]:
            pre = path2_prompt[-1]
            if data[d][0]['adjacency'][0][i][pre] > 0:
                path2_prompt.append('<su1>')
                path2_prompt.append(i)
            elif data[d][0]['adjacency'][0][pre][i] > 0:
                path2_prompt.append('<su2>')
                path2_prompt.append(i)
            elif data[d][0]['adjacency'][1][i][pre] > 0:
                path2_prompt.append('<te1>')
                path2_prompt.append(i)
            elif data[d][0]['adjacency'][1][pre][i] > 0:
                path2_prompt.append('<te2>')
                path2_prompt.append(i)
            elif data[d][0]['adjacency'][2][i][pre] > 0:
                path2_prompt.append('<ca1>')
                path2_prompt.append(i)
            elif data[d][0]['adjacency'][2][pre][i] > 0:
                path2_prompt.append('<ca2>')
                path2_prompt.append(i)
            elif data[d][0]['adjacency'][3][i][pre] > 0 or data[d][0]['adjacency'][3][pre][i] > 0:
                path2_prompt.append('<co1>')
                path2_prompt.append(i)
        for i in range(len(path2_prompt)):
            if i % 2 == 0:
                if i == 0:
                    path2_prompt[i] = ' ' + data[d][0]['node'][data[d][0]['ids_to_events'][path2_prompt[i]]]['mention']
                else:
                    path2_prompt[i] = data[d][0]['node'][data[d][0]['ids_to_events'][path2_prompt[i]]]['mention']

        data[d][0]['path1'] = ' '.join(path1_prompt)
        data[d][0]['path2'] = ' '.join(path2_prompt)

    return data


def get_dataloader(args):
    # 修改缓存文件名以区分ERNIE和BERT
    cache_file = 'ernie_data_cache.pth'  # 改为ERNIE专用缓存文件名

    if os.path.exists(cache_file):
        print("Loading ERNIE data from cache...")
        cached_data = torch.load(cache_file)

        # 重新初始化tokenizer并添加特殊token，确保词汇表大小一致
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)

        # 重新添加特殊token - 使用ERNIE格式的特殊token
        additional_mask = ['<ca0>', '<ca1>', '<ca2>', '<te0>', '<te1>', '<te2>',
                           '<co0>', '<co1>', '<su0>', '<su1>', '<su2>']  # 保持ERNIE格式
        c = AddedToken("<c>", rstrip=False, lstrip=True, single_word=False, normalized=True)  # 保持ERNIE格式
        for i in range(len(additional_mask)):
            additional_mask[i] = AddedToken(additional_mask[i], rstrip=False, lstrip=True, single_word=False,
                                            normalized=True)
        tokenizer.add_special_tokens(special_tokens_dict={'additional_special_tokens': additional_mask + [c]})

        # 重新添加多token事件标记
        to_add = cached_data['to_add']
        special_multi_event_token = list(to_add.keys())
        tokenizer.add_tokens(special_multi_event_token)
        args.vocab_size = len(tokenizer)

        # 保存ERNIE处理后的数据
        if hasattr(args, 'save_ernie_data') and args.save_ernie_data:
            save_ernie_data_processed(cached_data, tokenizer, to_add)

        return to_add, tokenizer, cached_data['train_dataloader'], cached_data['dev_dataloader'], cached_data[
            'test_dataloader']

    else:
        train_data = load_json(args.train_data_path)
        valid_data = load_json(args.valid_data_path)
        test_data = load_json(args.test_data_path)

        # train_data = train_data[:5]
        # valid_data = valid_data[:5]
        # test_data = test_data[:5]

        tokenizer = AutoTokenizer.from_pretrained(args.model_name)

        # 修改：更新schema生成函数中的特殊标记
        train_data = get_mention_schema(train_data, tokenizer, args)
        valid_data = get_mention_schema(valid_data, tokenizer, args)
        test_data = get_mention_schema(test_data, tokenizer, args)

        train_data = get_type_schema(train_data, tokenizer, args)
        valid_data = get_type_schema(valid_data, tokenizer, args)
        test_data = get_type_schema(test_data, tokenizer, args)

        train_data = get_common_prompt(train_data, tokenizer, args)
        valid_data = get_common_prompt(valid_data, tokenizer, args)
        test_data = get_common_prompt(test_data, tokenizer, args)

        train_data = get_core_event_prompt(train_data, tokenizer, args)
        valid_data = get_core_event_prompt(valid_data, tokenizer, args)
        test_data = get_core_event_prompt(test_data, tokenizer, args)

        train_data = get_document_prompt(train_data, tokenizer, args)
        valid_data = get_document_prompt(valid_data, tokenizer, args)
        test_data = get_document_prompt(test_data, tokenizer, args)

        # 收集多token事件
        multi_event, special_multi_event_token, event_dict, reverse_event_dict, to_add = collect_mult_event(
            train_data + valid_data + test_data, tokenizer)

        # 修改特殊token为ERNIE格式（保持尖括号）
        additional_mask = ['<ca0>', '<ca1>', '<ca2>',  # 因果关系
                           '<te0>', '<te1>', '<te2>',  # 时间关系
                           '<co0>', '<co1>',  # 共指关系
                           '<su0>', '<su1>', '<su2>']  # 子事件关系

        c = AddedToken("<c>", rstrip=False, lstrip=True, single_word=False, normalized=True)  # 事件标记
        for i in range(len(additional_mask)):
            additional_mask[i] = AddedToken(additional_mask[i], rstrip=False, lstrip=True, single_word=False,
                                            normalized=True)

        tokenizer.add_special_tokens(special_tokens_dict={'additional_special_tokens': additional_mask + [c]})

        tokenizer.add_tokens(special_multi_event_token)
        args.vocab_size = len(tokenizer)

        train_data = modify_sentences(train_data)
        valid_data = modify_sentences(valid_data)
        test_data = modify_sentences(test_data)

        # 多token事件替换
        train_data = replace_mult_event(train_data, reverse_event_dict)
        valid_data = replace_mult_event(valid_data, reverse_event_dict)
        test_data = replace_mult_event(test_data, reverse_event_dict)

        train_data = insert_event_marks(train_data, mark='<c>')  # 保持ERNIE格式
        valid_data = insert_event_marks(valid_data, mark='<c>')
        test_data = insert_event_marks(test_data, mark='<c>')

        train_data = get_key_schema(train_data, tokenizer, args)
        valid_data = get_key_schema(valid_data, tokenizer, args)
        test_data = get_key_schema(test_data, tokenizer, args)

        train_data = simplify_data(train_data, args, tokenizer)
        valid_data = simplify_data(valid_data, args, tokenizer)
        test_data = simplify_data(test_data, args, tokenizer)

        train_set = MyDataset(train_data, tokenizer, args)
        valid_set = MyDataset(valid_data, tokenizer, args)
        test_set = MyDataset(test_data, tokenizer, args)

        # 创建Dataloader，设置批大小和是否打乱顺序
        train_dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        dev_dataloader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False)
        test_dataloader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

        # 保存缓存数据
        cached_data = {
            'to_add': to_add,
            'train_dataloader': train_dataloader,
            'dev_dataloader': dev_dataloader,
            'test_dataloader': test_dataloader
        }
        torch.save(cached_data, cache_file)

        # 保存ERNIE处理后的数据
        if hasattr(args, 'save_ernie_data') and args.save_ernie_data:
            save_ernie_data_processed(cached_data, tokenizer, to_add)

        return to_add, tokenizer, train_dataloader, dev_dataloader, test_dataloader


def save_ernie_data_processed(cached_data, tokenizer, to_add):
    """
    保存ERNIE处理后的数据
    """
    import pickle
    import json

    # 创建保存目录
    ernie_data_dir = 'ernie_processed_data'
    if not os.path.exists(ernie_data_dir):
        os.makedirs(ernie_data_dir)

    # 保存tokenizer信息
    tokenizer_info = {
        'vocab_size': len(tokenizer),
        'special_tokens': tokenizer.all_special_tokens,
        'model_name': 'nghuyong/ernie-2.0-en',  # 改为ERNIE模型名称
        'to_add_keys': list(to_add.keys())
    }

    with open(f'{ernie_data_dir}/tokenizer_info.json', 'w') as f:
        json.dump(tokenizer_info, f, indent=2)

    # 保存多token事件信息
    with open(f'{ernie_data_dir}/multi_event_info.json', 'w') as f:
        json.dump(to_add, f, indent=2)

    # 保存完整的数据信息
    data_info = {
        'cache_file': 'ernie_data_cache.pth',
        'description': 'ERNIE processed data for event relation extraction'
    }

    with open(f'{ernie_data_dir}/data_info.json', 'w') as f:
        json.dump(data_info, f, indent=2)

    print(f"ERNIE processed data saved to {ernie_data_dir}/")

    # 保存简化的数据信息用于快速检查
    simplified_info = {
        'vocab_size': len(tokenizer),
        'special_tokens_count': len(tokenizer.all_special_tokens),
        'multi_event_tokens': len(to_add),
        'data_saved': True
    }

    with open(f'{ernie_data_dir}/simplified_info.json', 'w') as f:
        json.dump(simplified_info, f, indent=2)