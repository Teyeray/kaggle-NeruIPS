def train_graph_epoch(model, loader, optimizer):
    model.train()
    total = 0
    for data in loader:
        data = data.to(device)
        pred = model(data.x, data.edge_index, data.edge_attr, data.batch)
        loss = F.mse_loss(pred.view(-1), data.pseudo_weight)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        total += loss.item() * data.num_graphs
    return total / len(loader.dataset)

def objective_stage2(trial):
    # 超参空间
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-2)
    hidden_dim = trial.suggest_categorical("hidden_dim", [128, 256, 512])
    num_edge_layers = trial.suggest_int("num_edge_layers", 2, 5)
    
    # 加载最佳 Stage1 encoder
    encoder = WDMPNN(node_feat_dim, edge_feat_dim, hidden_dim,
                     num_edge_layers=num_edge_layers).to(device)
    encoder.load_state_dict(torch.load(f"stage1_best_enc_trial{study1.best_trial.number}.pt"))
    
    model = GraphLevelSSLModel(encoder).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(dataset_graph, batch_size=64, shuffle=True)
    
    best_loss = float("inf")
    for epoch in range(5):
        loss = train_graph_epoch(model, loader, optimizer)
        trial.report(loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        best_loss = min(best_loss, loss)
    
    torch.save(model.state_dict(), f"stage2_best_model_trial{trial.number}.pt")
    return best_loss

study2 = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())
study2.optimize(objective_stage2, n_trials=20)
print("Stage2 best params:", study2.best_trial.params)