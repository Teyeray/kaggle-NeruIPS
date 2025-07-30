def train_finetune_epoch(model, loader, optimizer):
    model.train()
    total = 0
    for data in loader:
        data = data.to(device)
        pred = model(data.x, data.edge_index, data.edge_attr, data.batch)
        loss = F.mse_loss(pred.view(-1), data.y)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        total += loss.item() * data.num_graphs
    return total / len(loader.dataset)

def objective_stage3(trial):
    # 超参空间
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-3)
    hidden_dim = trial.suggest_categorical("hidden_dim", [256, 512, 1024])
    # 迁移策略可选
    strategy = trial.suggest_categorical("transfer", ["a", "b", "c"])
    
    # 加载最佳 Stage2 模型
    full = FullModel(...).to(device)
    full.load_state_dict(torch.load(f"stage2_best_model_trial{study2.best_trial.number}.pt"))
    
    # 根据策略 a/b/c 决定冻结哪些参数
    if strategy == "a":
        for name, p in full.named_parameters():
            if "encoder" in name:
                p.requires_grad = False
    elif strategy == "b":
        # 冻结 encoder 但保留 predictor 前两层
        # … 自行实现 …
        pass
    # c：不冻结，全量微调
    
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, full.parameters()), lr=lr)
    train_loader = DataLoader(dataset_finetune_train, batch_size=32, shuffle=True)
    val_loader   = DataLoader(dataset_finetune_val,   batch_size=64, shuffle=False)
    
    best_val = float("inf")
    for epoch in range(10):
        train_loss = train_finetune_epoch(full, train_loader, optimizer)
        # 验证
        full.eval()
        val_loss = 0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                p = full(data.x, data.edge_index, data.edge_attr, data.batch)
                val_loss += F.mse_loss(p.view(-1), data.y, reduction="sum").item()
        val_loss /= len(val_loader.dataset)
        
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        best_val = min(best_val, val_loss)
    
    torch.save(full.state_dict(), f"stage3_best_model_trial{trial.number}.pt")
    return best_val

study3 = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())
study3.optimize(objective_stage3, n_trials=30)
print("Stage3 best params:", study3.best_trial.params)