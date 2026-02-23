import os

import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List

import vllm
from vllm import LLM, SamplingParams
from vllm.outputs import RequestOutput
from vllm.distributed.parallel_state import destroy_model_parallel

from transformers import AutoTokenizer
import datasets
import torch
import gc

os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1, 2, 3'

teacher_path = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(teacher_path)

def generate_prompt(prompt):
    
    global tokenizer
    messages = [
        {"role": "user", 
         "content": prompt}
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    return text

if __name__ == "__main__":
    
    gpu_num = 4
    max_token = 8192
    
    model = LLM(
        model=teacher_path,
        tensor_parallel_size=gpu_num,
        max_model_len=max_token,
        trust_remote_code=True,
        gpu_memory_utilization=0.95,
        dtype='auto',
        enforce_eager=True
    )

    sampling_params = SamplingParams(
        temperature=0.5, 
        top_p=0.8, 
        top_k=5,
        repetition_penalty=1.05, 
        max_tokens=2048
        )
    
    print('MODEL IS PREPARED')
    
    data = datasets.load_dataset('rosejae/alpaca_synthetic_prompt')
    df = pd.DataFrame(data['train'])
    df = df[:10000] 
    df.reset_index(drop=True)    
    
    prompts = df['prompt'].tolist()
    prompts = list(map(generate_prompt, prompts))
    
    batch_size = 10000
    generated_prompts = []
    generated_texts = []

    for batch_start in tqdm(range(0, len(prompts), batch_size)):
        batch_prompts = prompts[batch_start:batch_start + batch_size]

        outputs: List[RequestOutput] = model.generate(batch_prompts, sampling_params)

        batch_extracted_prompts = [output.prompt for output in outputs]
        batch_extracted_generated_texts = [output.outputs[0].text for output in outputs]
        
        generated_prompts.extend(batch_extracted_prompts)
        generated_texts.extend(batch_extracted_generated_texts)
        torch.cuda.empty_cache()

    df['generated'] = generated_texts
    df.to_csv('generate_iter1_alpaca_based.csv', index=False)
    
    iteration = 2
    
    while True:
        invalid_df = df[
        ~df['generated'].str.contains("Transformed Domain Question", na=False) |
        ~df['generated'].str.contains("Transformed Domain Answer", na=False) |
        ~df['generated'].str.contains("The answer is", na=False)]

        valid_df = df.drop(invalid_df.index)

        if invalid_df.empty:
            print(f"All rows contain 'Transformed Domain Question', 'Transformed Domain Answer', and 'The answer is'. Stopping after {iteration} iterations.")
            break

        if iteration > 1:
            valid_df.to_csv('valid_tmp_alpac.csv', index=False)
            invalid_df.to_csv('invalid_tmp_alpaca.csv', index=False)
            df.to_csv('tmp_concat_alpaca.csv', index=False)
            
        invalid_df = invalid_df.reset_index(drop=True)
        valid_df = valid_df.reset_index(drop=True)
        
        prompts = invalid_df['prompt'].tolist()
        prompts = list(map(generate_prompt, prompts))
        batch_size = 10000

        generated_prompts = []
        generated_texts = []

        for batch_start in tqdm(range(0, len(prompts), batch_size)):
            batch_prompts = prompts[batch_start:batch_start + batch_size]
            
            outputs: List[RequestOutput] = model.generate(batch_prompts, sampling_params)
            
            batch_extracted_prompts = [output.prompt for output in outputs]
            batch_extracted_generated_texts = [output.outputs[0].text for output in outputs]
            
            generated_prompts.extend(batch_extracted_prompts)
            generated_texts.extend(batch_extracted_generated_texts)
            torch.cuda.empty_cache()

        invalid_df['generated'] = generated_texts

        df = pd.concat([valid_df, invalid_df], axis=0)
        df = df.reset_index(drop=True)

    df.to_csv('final_generated_llama_alpaca.csv', index=False)

    destroy_model_parallel()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    torch.distributed.destroy_process_group()
    print("Successfully deleted the LLM pipeline and freed the GPU memory!")