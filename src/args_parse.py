import argparse

def create_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dataset_type", type=str, default='clean')

    parser.add_argument("--num_experts", type=int, default=4) #
    parser.add_argument("--topk", type=int, default=1) #
    parser.add_argument("--expert_dk", type=int, default=64) #
    parser.add_argument("--add_lr", type=float, default=0.0001) #
    parser.add_argument("--add_epoch", type=int, default=20)
    parser.add_argument("--semble_batch", type=int, default=16) #
    parser.add_argument("--model_expert_dk", type=int, default=5) #
    parser.add_argument("--model_topk", type=int, default=1) #
    parser.add_argument("--model_num_experts", type=int, default=10) #
    parser.add_argument("--model_moe_type", type=str, default=None) #
    parser.add_argument("--add_path_gap0", type=str, default=None) #
    parser.add_argument("--add_path_gap1", type=str, default=None)#
    parser.add_argument("--add_path_gap2", type=str, default=None)#
    parser.add_argument('--gap_type', type=int, choices=[0,1,2], default=0) #
    parser.add_argument("--connect_token", type=int, default=4 ) #
    parser.add_argument("--trans_drop", type=float, default=0.1) #
    parser.add_argument("--plm_lr", type=float, default=2e-5) #
    parser.add_argument('--model_name_or_path', type=str, default="models/RoBERTa-base-PM-M3-Voc-distill-align-hf/") #
    parser.add_argument("--model_type",type=str,default="roberta") #
    parser.add_argument("--chunk_size",type=int,default=128) #
    parser.add_argument("--best_model_path", type=str, default=None) #
    parser.add_argument("--use_different_lr",action="store_true") #
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1) #

    parser.add_argument('--data_dir', type=str, default="../data/mimicdata/mimiciii_clean/") #
    parser.add_argument('--label_titlefile', type=str,
                        default="../data/mimicdata/mimiciii_clean/split_lable_title_skipgram_full_1024.pkl") #
    parser.add_argument("--dataEnhance", action="store_true") #
    parser.add_argument("--dataEnhanceRatio", type=float, default=0.2,) #
    parser.add_argument("--is_trans", action="store_true") #
    parser.add_argument("--multiNum", type=int, default=4) #
    parser.add_argument("--dk", type=int, default=128) #
    parser.add_argument("--warmup", type=int, default=2) #
    parser.add_argument("--num_train_epochs", type=int, default=20) #
    parser.add_argument("--hidden_size", type=int, default=5) #
    parser.add_argument('--batch_size', type=int, default=8) #
    parser.add_argument("--n_epoch", type=int, default=20) #
    parser.add_argument("--optimiser", type=str, choices=["adagrad", "adam", "sgd", "adadelta", "adamw"],default="adamw") #
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate") #
    parser.add_argument("--weight_decay", type=float, default=0) #
    parser.add_argument("--use_lr_scheduler", type=int, choices=[0, 1], default=1) #
    parser.add_argument("--max_seq_length", type=int, default=4096) #
    # parser.add_argument("--min_word_frequency", type=int, default=-1)
    parser.add_argument("--attention_mode", type=str, choices=["text_label", "laat", "caml"], default="text_label") #
    args = parser.parse_args()
    return args



