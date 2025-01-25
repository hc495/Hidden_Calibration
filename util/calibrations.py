# Official implementation of paper "Token-based Decision Criteria Are Suboptimal in In-context Learning"
# Description: This file contains the calibration functions used in paper.
# @Author: Hakaze Cho, yfzhao@jaist.ac.jp
# @Date:   2025/01/24

import torch
import numpy as np
from tqdm import tqdm
import util.prompting as prompting

def softmax(x):
    f_x = np.exp(x) / np.sum(np.exp(x))
    return f_x

def pdf(x, mu, sigma): 
    return 1/(sigma*np.sqrt(2*np.pi))*np.exp(-(x-mu)**2/(2*sigma**2))

def empty_query_base_logits(
    model, 
    tokenizer, 
    dataset, 
    demos_amount, 
    prompt_function = None,
    total_tries=512, 
):
    if prompt_function is None:
        prompt_function = prompting.default_prompting
    output_distribution = []
    torch.cuda.empty_cache()
    prediction_count = [0] * len(dataset.label_space)
    with torch.no_grad():
        for i in tqdm(range(0, total_tries)):
            torch.cuda.empty_cache()
            if demos_amount == 0:
                prompt = ""
                true_label = ""
            else:
                prompt, true_label = prompt_function(dataset = dataset, demos_amount = demos_amount - 1, query_index=-1)
            fake_prompt = prompt + true_label + "\nInput: " + dataset.get_empty_input()[0] + ", Label: "
            tokenized_input = tokenizer(fake_prompt, return_tensors="pt").input_ids.cuda()
            result_vector = model(tokenized_input)['logits'][0][-1]
            label_space_p = []
            for labels in dataset.label_space:
                label_space_p.append(result_vector[tokenizer(labels, return_tensors="np").input_ids[0][-1]].cpu().detach().item())
            label_space_p = softmax(label_space_p)
            output_distribution.append(label_space_p)
            del(tokenized_input)

    output_distribution = np.array(output_distribution)

    return np.mean(output_distribution, axis=0).tolist()


def domain_text_base_logits(
    model, 
    tokenizer, 
    dataset, 
    demos_amount, 
    prompt_function = None,
    total_tries=512, 
    length=32,
):
    if prompt_function is None:
        prompt_function = prompting.default_prompting
    output_distribution = []
    torch.cuda.empty_cache()
    prediction_count = [0] * len(dataset.label_space)
    with torch.no_grad():
        for i in tqdm(range(0, total_tries)):
            torch.cuda.empty_cache()
            if demos_amount == 0:
                prompt = ""
                true_label = ""
            else:
                prompt, true_label = prompt_function(dataset = dataset, demos_amount = demos_amount - 1, query_index=-1)
            fake_prompt = prompt + true_label + "\nInput: " + dataset.get_domain_input(length) + ", Label: "
            tokenized_input = tokenizer(fake_prompt, return_tensors="pt").input_ids.cuda()
            result_vector = model(tokenized_input)['logits'][0][-1]
            label_space_p = []
            for labels in dataset.label_space:
                label_space_p.append(result_vector[tokenizer(labels, return_tensors="np").input_ids[0][-1]].cpu().detach().item())
            label_space_p = softmax(label_space_p)
            output_distribution.append(label_space_p)
            del(tokenized_input)

    output_distribution = np.array(output_distribution)

    return np.mean(output_distribution, axis=0).tolist()


def batch_calibration_for_result(
    results,
    batch_size
):
    ret = []
    step = len(results) // batch_size
    for i in range(step):
        batch = results[i * batch_size: (i + 1) * batch_size]
        mean_bias = np.mean(np.array(batch), axis=0)
        for j in range(batch_size):
            ret.append(np.argmax(np.array(batch[j]) - mean_bias))
    last_batch = results[step * batch_size:]
    mean_bias = np.mean(np.array(last_batch), axis=0)
    for j in range(len(last_batch)):
        ret.append(np.argmax(np.array(last_batch[j]) - mean_bias))
    return ret


def predict_by_knn(
    feature,
    example,
    examplelabel,
    distance_calculator,
    label_space_length,
    least_k = 3,
):
    distances = []
    knns = [0]*label_space_length
    for e in example:
        distance = distance_calculator(e, feature)
        distances.append(distance)
    distances = np.array(distances)
    sorted_index = np.argsort(distances)
    now_index = 0
    while (now_index < len(distances)) and (len(np.argwhere(knns == np.max(knns))) > 1) or (np.sum(knns) < least_k):
        knns[examplelabel[sorted_index[now_index]]] += 1
        now_index += 1
    return np.argmax(knns).item()