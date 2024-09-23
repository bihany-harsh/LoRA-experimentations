from collections import namedtuple
import pandas as pd
import pickle

ModelHistory = namedtuple("ModelHistory", ["trainable_params", "total_params", "results", "test_loss", "test_accuracy"])

with open("./out_files/MNIST_512_4_FFN_history_finetuned.pkl", "rb") as f:
    finetuned_histories = pickle.load(f)
with open("./out_files/MNIST_512_4_lora_experiment.pkl", "rb") as f:
    lora_histories = pickle.load(f)
with open("./out_files/MNIST_512_4_shift_lora_experiment.pkl", "rb") as f:
    shift_r_lora_histories = pickle.load(f)
with open("./out_files/MNIST_512_4_pre_mult_lora_experiment.pkl", "rb") as f:
    pre_mult_r_lora_histories = pickle.load(f)
with open("./out_files/MNIST_512_4_shift_pre_mult_lora_experiment.pkl", "rb") as f:
    shift_pre_mult_r_lora_histories = pickle.load(f)
with open("./out_files/MNIST_512_4_post_mult_lora_experiment.pkl", "rb") as f:
    post_mult_r_lora_histories = pickle.load(f)
with open("./out_files/MNIST_512_4_shift_post_mult_lora_experiment.pkl", "rb") as f:
    shift_post_mult_r_lora_histories = pickle.load(f)
with open("./out_files/MNIST_512_4_post_mult_lora_fix_experiment.pkl", "rb") as f:
    post_mult_r_lora_fix_histories = pickle.load(f)
with open("./out_files/MNIST_512_4_shift_post_mult_lora_fix_experiment.pkl", "rb") as f:
    shift_post_mult_r_lora_fix_histories = pickle.load(f)

    
data = {
    'rank': [],
    'params (Fine Tuning)': [],
    'params (LoRA r)': [],
    'params (Shift_LoRA r)': [],
    'params (Pre_Mult_LoRA r)': [],
    'params (Shift_Pre_Mult_LoRA r)': [],
    'params (Post_Mult_LoRA r)': [],
    'params (Shift_Post_Mult_LoRA r)': [],
    'params (Post_Mult_Fix_LoRA r)': [],
    'params (Shift_Post_Mult_Fix_LoRA r)': [],
}

ranks = [2**i for i in range(10)]

for rank in ranks:
    data['rank'].append(rank)
    data['params (Fine Tuning)'].append(finetuned_histories.trainable_params)
    data['params (LoRA r)'].append(lora_histories[rank].trainable_params)
    data['params (Shift_LoRA r)'].append(shift_r_lora_histories[rank].trainable_params)
    data['params (Pre_Mult_LoRA r)'].append(pre_mult_r_lora_histories[rank].trainable_params)
    data['params (Shift_Pre_Mult_LoRA r)'].append(shift_pre_mult_r_lora_histories[rank].trainable_params)
    data['params (Post_Mult_LoRA r)'].append(post_mult_r_lora_histories[rank].trainable_params)
    data['params (Shift_Post_Mult_LoRA r)'].append(shift_post_mult_r_lora_histories[rank].trainable_params)
    data['params (Post_Mult_Fix_LoRA r)'].append(post_mult_r_lora_fix_histories[rank].trainable_params)
    data['params (Shift_Post_Mult_Fix_LoRA r)'].append(shift_post_mult_r_lora_fix_histories[rank].trainable_params)
    
df = pd.DataFrame(data)
df.to_csv("results.csv")