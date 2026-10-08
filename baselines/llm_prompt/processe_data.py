# -*- coding: utf-8 -*-#
import copy

import tqdm
import torch
import random
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer

from data_set import MyDataset
from load_data import load_json
from tokenizers import AddedToken


# -------------------------------------------------------------------------------
# Name:         processe-data
# Description:
# Author:       梁超
# Date:         2024/5/16
# -------------------------------------------------------------------------------
def get_mention_schema(data, args):
    process = tqdm.tqdm(total=len(data), ncols=75, desc='Mention schema')
    for d in range(len(data)):
        mention_schema = []
        process.update(1)
        sub_r = [[i[0], i[1], 'Include'] if random.random() > 0.5 else [i[1], i[0], 'Belong to'] for i in data[d][0]['relation']['SUB_EVENT']]
        temp_r = [[i[0], i[1], 'Before'] if random.random() > 0.5 else [i[1], i[0], 'After'] for i in data[d][0]['relation']['TEMPORAL']]
        cau_r = [[i[0], i[1], 'Cause'] if random.random() > 0.5 else [i[1], i[0], 'Caused by'] for i in data[d][0]['relation']['CAUSAL']]
        cof_r = [[i[0], i[1], 'Cof'] for i in data[d][0]['relation']['COF_EVENT']]

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

        # if len(mention_schema) >= 80:
        #     list_del = []
        #     for i in mention_schema:
        #         if i[0].split('_')[0] == 'COFM' and int(i[0].split('_')[1]) >= 2 and i[0] != e1 and i[0] != e2:
        #             list_del.append(i)
        #         elif i[1].split('_')[0] == 'COFM' and int(i[1].split('_')[1]) >= 2 and i[1] != e1 and i[1] != e2:
        #             list_del.append(i)
        #     for i in list_del:
        #         mention_schema.remove(i)

        mention_schema = [data[d][0]['node'][j[0]]['mention'] + ' ' + j[2] + ' ' + data[d][0]['node'][j[1]]['mention'] for j in mention_schema]
        mention_schema.reverse()
        data[d][0]['mention_schema'] = ' </s> '.join(mention_schema) + ' </s> '
        # while len(tokenizer.encode(data[d][0]['mention_schema'])) >= args.len_schema - 12:
        #     l = len(mention_schema)
        #     mention_schema = mention_schema[int(0.2 * l):]
        #     data[d][0]['mention_schema'] = ' </s> '.join(mention_schema) + ' </s> ' + data[d][0]['node'][e1]['mention'] + ' <mask> ' + data[d][0]['node'][e2]['mention']
    return data

def get_type_schema(data, tokenizer, args):
    origin_len = []
    determine_len = []

    process = tqdm.tqdm(total=len(data), ncols=75, desc='Type schema')
    for d in range(len(data)):
        type_schema = []
        process.update(1)
        sub_r = [[i[0], i[1], '<su1>'] if random.random() > 0.5 else [i[1], i[0], '<su2>'] for i in data[d][0]['relation']['SUB_EVENT']]
        temp_r = [[i[0], i[1], '<te1>'] if random.random() > 0.5 else [i[1], i[0], '<te2>'] for i in data[d][0]['relation']['TEMPORAL']]
        cau_r = [[i[0], i[1], '<ca1>'] if random.random() > 0.5 else [i[1], i[0], '<ca2>'] for i in data[d][0]['relation']['CAUSAL']]
        cof_r = [[i[0], i[1], '<co1>'] for i in data[d][0]['relation']['COF_EVENT']]

        all_relations = sub_r + temp_r + cau_r + cof_r
        for i in range(len(all_relations)):
            all_relations[i] = [data[d][0]['node'][all_relations[i][0]]['type'], data[d][0]['node'][all_relations[i][1]]['type'], all_relations[i][2]]

        if args.diff_type == 1:  # 如果args.diff_type为1，就将all_relations中重复的三元组删去
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
        data[d][0]['type_schema'] = ' </s> '.join(type_schema) + ' </s> ' + e1 + ' <mask> ' + e2
        origin_len.append(len(type_schema))
        while len(tokenizer.encode(data[d][0]['type_schema'])) >= args.len_schema - 12:
            l = len(type_schema)
            type_schema = type_schema[int(0.2*l):]
            data[d][0]['type_schema'] = ' </s> '.join(type_schema) + ' </s> ' + e1 + ' <mask> ' + e2
        determine_len.append(len(type_schema))
    return data

def modify_sentences(data):
    process = tqdm.tqdm(total=len(data), ncols=75, desc='Modify')
    for d in range(len(data)):
        process.update(1)
        sentence_lists = {}
        for i in data[d][0]['node'].keys():
            sent_id = data[d][0]['node'][i]['sent_id']
            sentence_lists[str(sent_id)] = data[d][0]['node'][i]['sentence']
        data[d][0]['sentences'] = sentence_lists
    return data

def collect_mult_event(data, tokenizer):
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
    process = tqdm.tqdm(total=len(data), ncols=75, desc='Replacing')
    for d in range(len(data)):
        process.update(1)

        event_id_list = list(data[d][0]['node'].keys())
        sorted_event_ids = sorted(event_id_list, key=lambda x: (1000 * data[d][0]['node'][x]['sent_id'] + data[d][0]['node'][x]['location'][0]), reverse=True)

        for event_id in sorted_event_ids:
            sent_id = data[d][0]['node'][event_id]['sent_id']
            mention = data[d][0]['node'][event_id]['mention']
            sentence = data[d][0]['sentences'][str(sent_id)]
            location = data[d][0]['node'][event_id]['location']

            if mention in reverse_event_dict:
                data[d][0]['node'][event_id]['mention'] = reverse_event_dict[mention]
                data[d][0]['sentences'][str(sent_id)] = sentence[:location[0]] + [reverse_event_dict[mention]] + sentence[location[1]:]
                data[d][0]['node'][event_id]['location'] = [location[0], location[0] + 1]

    process.close()
    return data


def insert_event_marks(data, mark='<c>'):
    process = tqdm.tqdm(total=len(data), ncols=75, desc='Inserting')
    for d in range(len(data)):
        process.update(1)

        event_id_list = list(data[d][0]['node'].keys())
        sorted_event_ids = sorted(event_id_list, key=lambda x: (1000 * data[d][0]['node'][x]['sent_id'] + data[d][0]['node'][x]['location'][0]), reverse=True)

        pre_sen_id = data[d][0]['node'][sorted_event_ids[0]]['sent_id']
        sent_loc = 0
        for event_id in sorted_event_ids:
            cur_sen_id = data[d][0]['node'][event_id]['sent_id']
            if cur_sen_id != pre_sen_id:
                sent_loc = 0

            mention = data[d][0]['node'][event_id]['mention']
            sentence = data[d][0]['sentences'][str(cur_sen_id)]
            location = data[d][0]['node'][event_id]['location']

            data[d][0]['sentences'][str(cur_sen_id)] = sentence[:location[0]] + [mark, mention, mark] + sentence[location[1]:]
            data[d][0]['node'][event_id]['location'] = sent_loc
            sent_loc += 1
            pre_sen_id = cur_sen_id
    return data

def simplify_data(data, args, tokenizer):
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
            candi_loc = torch.nonzero(idx == 50276, as_tuple=False)[:, 1][::2]

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
            locs.append(int(cands[l-loc]) + 1)

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
        event_mention_loc = torch.nonzero(idx_event_schema == 50264, as_tuple=False)[0][1]

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
        event_type_loc = torch.nonzero(idx_type_schema == 50264, as_tuple=False)[0][1]

        n = {'idx': idxs,
             'mask': masks,
             'sentence_ids': sent_ids,
             'location': locs,
             'event_schema': idx_event_schema,
             'event_schema_mask': mask_event_schema,
             'event_mention_loc': event_mention_loc,
             'event_type_loc': event_type_loc,
             'type_schema': idx_type_schema,
             'type_schema_mask': mask_type_schema
             }

        data[d] = [n, adjacency_matrix, [events_to_ids[data[d][1][0]], events_to_ids[data[d][1][1]]], data[d][2]]
    return data


def get_dataloader(args):
    train_data = load_json(args.train_data_path)
    valid_data = load_json(args.valid_data_path)
    test_data = load_json(args.test_data_path)

    # train_data = train_data[:10]
    # valid_data = valid_data[:10]
    # test_data = test_data[:10]

    tokenizer = RobertaTokenizer.from_pretrained(args.model_name)

    train_data = get_mention_schema(train_data, tokenizer, args)
    valid_data = get_mention_schema(valid_data, tokenizer, args)
    test_data = get_mention_schema(test_data, tokenizer, args)

    train_data = get_type_schema(train_data, tokenizer, args)
    valid_data = get_type_schema(valid_data, tokenizer, args)
    test_data = get_type_schema(test_data, tokenizer, args)

    # 收集多token事件
    multi_event, special_multi_event_token, event_dict, reverse_event_dict, to_add = collect_mult_event(
        train_data + valid_data + test_data, tokenizer)

    additional_mask = ['<ca0>', '<ca1>', '<ca2>',   # 50265、50266、50267
                       '<te0>', '<te1>', '<te2>',   # 50268、50269、50270
                       '<co0>', '<co1>',            # 50271、50272
                       '<su0>', '<su1>', '<su2>']   # 50273、50274、50275

    tokenizer.add_tokens(additional_mask)

    c = AddedToken("<c>", rstrip=False, lstrip=True, single_word=False, normalized=True)  # 50276
    # SUB-TEMP-CAU-COF

    tokenizer.add_special_tokens(special_tokens_dict={'additional_special_tokens': [c]})

    tokenizer.add_tokens(special_multi_event_token)  #
    args.vocab_size = len(tokenizer)  # 50265+8+

    train_data = modify_sentences(train_data)
    valid_data = modify_sentences(valid_data)
    test_data = modify_sentences(test_data)

    # 多token事件替换
    train_data = replace_mult_event(train_data, reverse_event_dict)
    valid_data = replace_mult_event(valid_data, reverse_event_dict)
    test_data = replace_mult_event(test_data, reverse_event_dict)


    train_data = insert_event_marks(train_data, mark='<c>')
    valid_data = insert_event_marks(valid_data, mark='<c>')
    test_data = insert_event_marks(test_data, mark='<c>')

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
    return to_add, tokenizer, train_dataloader, dev_dataloader, test_dataloader



def get_dataloader_for_test(args):
    train_data = load_json(args.train_data_path)
    valid_data = load_json(args.valid_data_path)
    test_data = load_json(args.test_data_path)

    # train_data = train_data[:10]
    # valid_data = valid_data[:10]
    # test_data = test_data[:10]

    tokenizer = RobertaTokenizer.from_pretrained(args.model_name)

    test_data = get_mention_schema(test_data, tokenizer, args)

    test_data = get_type_schema(test_data, tokenizer, args)

    # 收集多token事件
    multi_event, special_multi_event_token, event_dict, reverse_event_dict, to_add = collect_mult_event(
        train_data + valid_data + test_data, tokenizer)

    additional_mask = ['<ca0>', '<ca1>', '<ca2>',   # 50265、50266、50267
                       '<te0>', '<te1>', '<te2>',   # 50268、50269、50270
                       '<co0>', '<co1>',            # 50271、50272
                       '<su0>', '<su1>', '<su2>']   # 50273、50274、50275

    tokenizer.add_tokens(additional_mask)

    c = AddedToken("<c>", rstrip=False, lstrip=True, single_word=False, normalized=True)  # 50276
    # SUB-TEMP-CAU-COF

    tokenizer.add_special_tokens(special_tokens_dict={'additional_special_tokens': [c]})

    tokenizer.add_tokens(special_multi_event_token)  #
    args.vocab_size = len(tokenizer)  # 50265+8+

    test_data = modify_sentences(test_data)

    # 多token事件替换
    test_data = replace_mult_event(test_data, reverse_event_dict)

    test_data = insert_event_marks(test_data, mark='<c>')

    test_data = simplify_data(test_data, args, tokenizer)

    test_set = MyDataset(test_data, tokenizer, args)


    # 创建Dataloader，设置批大小和是否打乱顺序
    test_dataloader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    return to_add, tokenizer, test_dataloader


def trans_llm(data, args):
    llm_input = []
    labels = ['Sub-event', 'Temporal', 'Causal', 'Coreference']
    cads = [['Belong to', 'Include', 'None'], ['Before', 'After', 'None'], ['Cause', 'Caused by', 'None'], ['Cof', 'None']]

    if args.single == 1:
        ques = [[' " means Event1 is a sub-event of Event2 , while " ', ' " means Event2 is a sub-event of Event1 ) . If there is no such relationship, respond with " None " . Please respond with a single word or phrase .'],
                [' " means Event1 occurs before Event2 in time , while " ', ' " means Event2 occurs before Event1 in time ) . If there is no such relationship, respond with " None " . Please respond with a single word or phrase .'],
                [' " means Event1 is the cause of Event2 , while " ', ' " means Event2 is the cause of  Event1 ) . If there is no such relationship, respond with " None " . Please respond with a single word or phrase .'],
                [' " means Event1 and Event2 refer to the same event . ) ', 'If there is no such relationship, respond with " None " . Please respond with a single word or phrase .']]
    else:
        ques = [[' " means Event1 is a sub-event of Event2 , while " ', ' " means Event2 is a sub-event of Event1 ) . If there is no such relationship, respond with " None " .'],
                [' " means Event1 occurs before Event2 in time , while " ', ' " means Event2 occurs before Event1 in time ) . If there is no such relationship, respond with " None " .'],
                [' " means Event1 is the cause of Event2 , while " ', ' " means Event2 is the cause of  Event1 ) . If there is no such relationship, respond with " None " .'],
                [' " means Event1 and Event2 refer to the same event . ) ', 'If there is no such relationship, respond with " None " .']]
    for i in range(len(data)):
        for p in range(len(labels)):
            temp = {}
            temp['Description'] = 'We annotated the events and their relationships in the given document , and constructed an event graph with the events as nodes and the relationships between them as edges . There are four types of relations: Sub-event, Temporal, Causal, Coreference .'
            temp['Candidates'] = ', '.join(cads[p])
            sentences = []
            id_sentence = {}
            max_sent = 0
            for j in data[i][0]['node'].keys():
                id_sentence[j] = data[i][0]['node'][j]['sent_id']
                max_sent = max(max_sent, id_sentence[j])

            for pp in range(max_sent + 1):
                all_events = []
                for j in data[i][0]['node'].keys():
                    if data[i][0]['node'][j]['sent_id'] == pp:
                        all_events.append(data[i][0]['node'][j]['mention'])
                if len(all_events) > 0:
                    all_events = ' , '.join(all_events) + ' : ' + data[i][0]['sentences'][pp]
                    sentences.append(all_events)

            temp['sentences'] = ' ; '.join(sentences)
            temp['structure'] = data[i][0]['mention_schema']

            temp['Question'] = 'Please determine if there is a ' + labels[p] + ' relationship between Event1 ' + data[i][0]['node'][data[i][1][0]]['mention'] + ' and Event2 ' + data[i][0]['node'][data[i][1][1]]['mention'] + '. If there is, respond with " ' + cads[p][0] + ' " or " ' + cads[p][1] + ' " ( " ' + cads[p][0] + ques[p][0] + cads[p][1] + ques[p][1]
            if labels[p] == 'Coreference':
                temp['Question'] = 'Please determine if there is a ' + labels[p] + ' relationship between Event1 ' + data[i][0]['node'][data[i][1][0]]['mention'] + ' and Event2 ' + data[i][0]['node'][data[i][1][1]]['mention'] + '. If there is, respond with " ' + cads[p][0] + ' " ( " ' + cads[p][0] + ques[p][0]

            input_str = ''
            for k in temp.keys():
                input_str += k + ' : ' + temp[k] + '\n'
            llm_input.append([input_str, cads[p][data[i][2][p]]])

    return llm_input


