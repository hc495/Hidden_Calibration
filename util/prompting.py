# Official implementation of paper "Token-based Decision Criteria Are Suboptimal in In-context Learning"
# Description: This file contains the input formatting functions used in paper.
# @Author: Hakaze Cho, yfzhao@jaist.ac.jp
# @Date:   2025/01/24

import random

def default_prompting(
        dataset, 
        demos_amount, 
        query_index = -1,
        input_start = 'Input: ',
        input_end_label_start = ', Label: ',
        inter_div = '\n',
        pre_sampled_demonstration = None
    ):
    if pre_sampled_demonstration is not None:
        sample_list = pre_sampled_demonstration
    else:
        sample_list = []
        if query_index == -1:
            sample_list = random.sample(range(0, dataset.get_max()), demos_amount + 1)
        else:
            sample_list = random.sample(range(0, dataset.get_max()), demos_amount)
            if query_index in sample_list:
                sample_list = random.sample(range(0, dataset.get_max()), demos_amount)
    sample_list.append(query_index)
    
    prompt = ''
    query_label = 0
    
    for i in range(0, demos_amount):
        text, label = dataset.get(sample_list[i])
        prompt += input_start
        prompt += text
        prompt += input_end_label_start
        prompt += label
        prompt += inter_div
    text, label = dataset.get(sample_list[-1])
    prompt += input_start
    prompt += text
    prompt += input_end_label_start
    query_label = label

    return prompt, query_label


def empty_prompting(
    dataset,
    demos_amount,
    query_index = 0
):
    prompt = ''
    text, label = dataset.get(query_index)
    prompt += text
    return prompt, label


def default_prompting_without_last_colon(
    dataset, 
    demos_amount, 
    query_index = -1
):
    sample_list = []
    if query_index == -1:
        sample_list = random.sample(range(0, dataset.get_max()), demos_amount + 1)
    else:
        sample_list = random.sample(range(0, dataset.get_max()), demos_amount)
        if query_index in sample_list:
            sample_list = random.sample(range(0, dataset.get_max()), demos_amount)
        sample_list.append(query_index)
    
    prompt = ''
    query_label = 0
    
    for i in range(0, demos_amount):
        text, label = dataset.get(sample_list[i])
        prompt += 'Input: '
        prompt += text
        prompt += ', Label: '
        prompt += label
        prompt += '\n'
    text, label = dataset.get(sample_list[-1])
    prompt += 'Input: '
    prompt += text
    prompt += ', Label'
    query_label = label

    return prompt, query_label


def default_prompting_with_random_label(
    dataset, 
    demos_amount, 
    query_index = -1
):
    sample_list = []
    if query_index == -1:
        sample_list = random.sample(range(0, dataset.get_max()), demos_amount + 1)
    else:
        sample_list = random.sample(range(0, dataset.get_max()), demos_amount)
        if query_index in sample_list:
            sample_list = random.sample(range(0, dataset.get_max()), demos_amount)
        sample_list.append(query_index)
    
    prompt = ''
    query_label = 0
    
    for i in range(0, demos_amount):
        text, _ = dataset.get(sample_list[i])
        label = random.choice(dataset.label_space)
        prompt += 'Input: '
        prompt += text
        prompt += ', Label: '
        prompt += label
        prompt += '\n'
    text, label = dataset.get(sample_list[-1])
    prompt += 'Input: '
    prompt += text
    prompt += ', Label: '
    query_label = label

    return prompt, query_label


def default_prompting_with_new_label_space(
    dataset, 
    demos_amount, 
    label_space,
    query_index = -1
):
    dataset.change_label_space(label_space)
    sample_list = []
    if query_index == -1:
        sample_list = random.sample(range(0, dataset.get_max()), demos_amount + 1)
    else:
        sample_list = random.sample(range(0, dataset.get_max()), demos_amount)
        if query_index in sample_list:
            sample_list = random.sample(range(0, dataset.get_max()), demos_amount)
        sample_list.append(query_index)
    
    prompt = ''
    query_label = 0
    
    for i in range(0, demos_amount):
        text, label = dataset.get(sample_list[i])
        prompt += 'Input: '
        prompt += text
        prompt += ', Label: '
        prompt += label
        prompt += '\n'
    text, label = dataset.get(sample_list[-1])
    prompt += 'Input: '
    prompt += text
    prompt += ', Label: '
    query_label = label

    return prompt, query_label


def default_prompting_with_difference_datasets(
    dataset, 
    test_dataset,
    demos_amount, 
    query_index = 0
):
    train_dataset = dataset
    sample_list = []
    sample_list = random.sample(range(0, train_dataset.get_max()), demos_amount)
    if query_index in sample_list:
        sample_list = random.sample(range(0, test_dataset.get_max()), demos_amount)
    sample_list.append(query_index)
    
    prompt = ''
    query_label = 0
    
    for i in range(0, demos_amount):
        text, label = train_dataset.get(sample_list[i])
        prompt += 'Input: '
        prompt += text
        prompt += ', Label: '
        prompt += label
        prompt += '\n'
    text, label = test_dataset.get(sample_list[-1])
    prompt += 'Input: '
    prompt += text
    prompt += ', Label: '
    query_label = label

    return prompt, query_label