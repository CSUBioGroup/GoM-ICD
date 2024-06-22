from datasets import load_dataset
import torch

def set_random_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_data(data_dir):
    # data_dir = args.data_dir
    # data_dir = "../data/mimicdata/mimic3/50"
    train_path=data_dir+"train.csv"
    valid_path=data_dir+"valid.csv"
    test_path=data_dir+"test.csv"
    data_files = {}
    data_files["train"] = train_path
    data_files["valid"] = valid_path
    data_files["test"] = test_path
    raw_datasets = load_dataset("csv", data_files=data_files)
    return raw_datasets



def reformat(code, is_diag):
    code = ''.join(code.split('.'))

    if is_diag:
        if code.startswith('E'):
            if len(code) > 4:
                code = 'diag_'+code[:4] + '.' + code[4:]
            else:
                code = 'diag_' + code
        else:
            if len(code) > 3:
                code = 'diag_'+code[:3] + '.' + code[3:]
            else:
                code = 'diag_' + code
    else:
        code = 'pro_'+code[:2] + '.' + code[2:]

    return code




