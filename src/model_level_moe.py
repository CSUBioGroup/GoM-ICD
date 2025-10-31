from run_src.run_rnn_moe_gap_icd import *

def run_model(args,vocab,label_vector,accelerator):
    if args.model_moe_type is None:
        pass
    elif args.model_moe_type=='sigmoid':
        label_vector['train_outputs_np']=torch.tensor(label_vector['train_outputs_np']).sigmoid().numpy()
        label_vector['val_outputs_np']=torch.tensor(label_vector['val_outputs_np']).sigmoid().numpy()
        label_vector['test_outputs_np']=torch.tensor(label_vector['test_outputs_np']).sigmoid().numpy()

    dataset = CustomDataset(label_vector['train_outputs_np'], label_vector['train_target_tensor'])
    train_moe_dataloader = DataLoader(dataset, batch_size=args.semble_batch, shuffle=True)
    dataset = CustomDataset(label_vector['val_outputs_np'], label_vector['val_target_tensor'])
    eval_semble_dataloader = DataLoader(dataset, batch_size=args.semble_batch)
    dataset = CustomDataset(label_vector['test_outputs_np'], label_vector['test_target_tensor'])
    test_semble_dataloader = DataLoader(dataset, batch_size=args.semble_batch)

    eval_semble_dataloader = accelerator.prepare(eval_semble_dataloader)
    test_semble_dataloader = accelerator.prepare(test_semble_dataloader)

    model_semble = MoE_model_level(args,vocab)
    betas = (0.9, 0.999)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model_semble.parameters()),
                      lr=args.add_lr, betas=betas)

    T_epochs = args.num_train_epochs
    if args.use_lr_scheduler:
        itersPerEpoch = len(train_moe_dataloader)
        epoch = T_epochs
        warmupEpochs = args.warmup
        lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmupEpochs * itersPerEpoch,
                                                       num_training_steps=epoch * itersPerEpoch)
    model_semble, optimizer, train_moe_dataloader = accelerator.prepare(
        model_semble, optimizer, train_moe_dataloader
    )
    criterions = nn.BCEWithLogitsLoss()
    num_update_steps_per_epoch = math.ceil(len(train_moe_dataloader))
    max_train_steps = args.n_epoch * num_update_steps_per_epoch
    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
    print(datetime.datetime.now().strftime('%H:%M:%S'))
    checkpoint = {}
    batch_id = 0
    metrics_max = None
    metrics_max_val = -1
    for epoch in tqdm(range(args.add_epoch)):
        model_semble.train()
        optimizer.zero_grad()
        losses = []
        epoch_loss = 0.0

        for step, batch in enumerate(train_moe_dataloader):
            batch_id += 1
            output = model_semble(batch[0])
            loss = criterions(output, batch[1])
            accelerator.backward(loss)
            losses.append(loss.item())
            epoch_loss += loss.item()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
            progress_bar.set_postfix(loss=epoch_loss / batch_id)

        model_semble.eval()
        all_preds = []
        all_preds_raw = []
        all_labels = []
        for step, batch in enumerate(eval_semble_dataloader):
            with torch.no_grad():
                outputs = model_semble(batch[0])
            preds_raw = outputs.sigmoid().cpu()
            preds = (preds_raw > 0.5).int()
            all_preds_raw.extend(list(preds_raw))
            all_preds.extend(list(preds))
            all_labels.extend(list(batch[1].cpu().numpy()))
        all_preds_raw = np.stack(all_preds_raw)
        all_preds = np.stack(all_preds)
        all_labels = np.stack(all_labels)
        metrics = all_metrics(yhat=all_preds, y=all_labels, yhat_raw=all_preds_raw)
        metrics_val = metrics
        print("lr{}:".format(optimizer.param_groups[0]["lr"]), "loss: ", np.mean(losses).item())
        print(
            f"F1_micro: {metrics['f1_micro']:.4f}, F1_macro: {metrics['f1_macro']:.4f}, auc_micro: {metrics['auc_micro']:.4f}, auc_macro: {metrics['auc_macro']:.4f}, prec_at_8: {metrics['prec_at_8']:.4f}")
        print(" ")

        metrics_test = metrics
        if metrics_max is None:
            metrics_max = metrics_test
            metrics_max_val = metrics_val['f1_micro']
            metrics_max_auc = metrics_val['auc_micro']
            checkpoint = {'the epoch': epoch,
                          'metrics': metrics_max,
                          'model_state_dict': model_semble.state_dict()}
            torch.save(checkpoint, args.best_model_path+'moe_best.pth')
        else:
            if metrics_max_val < metrics_val['f1_micro']:
                metrics_max_val = metrics_val['f1_micro']
                metrics_max_auc = metrics_val['auc_micro']
                metrics_max = metrics_test
                checkpoint = {'the epoch': epoch,
                              'metrics': metrics_max,
                              'model_state_dict': model_semble.state_dict()}
                torch.save(checkpoint, args.best_model_path+'moe_best.pth')

    print(datetime.datetime.now().strftime('%H:%M:%S'))
    checkpoint = torch.load(args.best_model_path+'moe_best.pth', map_location=device)
    model_semble.load_state_dict(checkpoint['model_state_dict'])

    model_semble.eval()
    all_preds_raw = []
    all_labels = []
    all_preds = []

    for step, batch in enumerate(test_semble_dataloader):
        with torch.no_grad():
            outputs = model_semble(batch[0])
        preds_raw = outputs.sigmoid().cpu()
        preds = (preds_raw > 0.5).int()
        all_preds_raw.extend(list(preds_raw))
        all_preds.extend(list(preds))
        all_labels.extend(list(batch[1].cpu().numpy()))
    all_preds_raw = np.stack(all_preds_raw)
    all_preds = np.stack(all_preds)
    all_labels = np.stack(all_labels)
    metrics = all_metrics(yhat=all_preds, y=all_labels, yhat_raw=all_preds_raw)
    list_threshold=[0.3, 0.35, 0.4, 0.45, 0.5]

    best_thre=0
    for t in list_threshold:
        all_preds = (all_preds_raw > t).astype(int)
        metrics = all_metrics(yhat=all_preds, y=all_labels, yhat_raw=all_preds_raw, k=[5, 8, 15])

    model_semble=None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

class MoE_model_level(nn.Module):
    def __init__(self,args,vocab):
        super(MoE_model_level, self).__init__()
        self.name="moe_model"
        self.moe_model_layer = SwitchMoE_model(dim=3, hidden_dim=args.model_expert_dk, num_experts=args.model_num_experts,
                                   topk_expert=args.model_topk)
        self.pre_linears = nn.Linear(3, 1, bias=True)
        self.third_linears =nn.Linear(3,vocab.label_num, bias=True)

    def forward(self, batch_data):
        output=batch_data
        output = self.moe_model_layer(batch_data)
        output=output.reshape(output.shape[0], -1)
        return output

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class SwitchMoE_model(nn.Module):

    def __init__(
            self,
            dim: int,
            hidden_dim: int,
            num_experts: int,
            capacity_factor: float = 1.0,
            mult: int = 4,
            topk_expert: int = 1,
            *args,
            **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.mult = mult
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.Dropout(p=0.5),
                nn.Linear(hidden_dim, 1)
            ) for _ in range(num_experts)
        ])
        self.gate = SwitchGate(
            dim,
            num_experts,
            capacity_factor,
            1e-6,
            topk_expert,
        )

    def forward(self, x: Tensor):

        gate_scores = self.gate(x)
        expert_outputs = [expert(x) for expert in self.experts]
        if torch.isnan(gate_scores).any():
            gate_scores[torch.isnan(gate_scores)] = 0

        stacked_expert_outputs = torch.stack(
            expert_outputs, dim=-1
        )
        if torch.isnan(stacked_expert_outputs).any():
            stacked_expert_outputs[
                torch.isnan(stacked_expert_outputs)
            ] = 0
        moe_output = torch.sum(
            gate_scores.unsqueeze(-2) * stacked_expert_outputs, dim=-1
        )
        return moe_output
