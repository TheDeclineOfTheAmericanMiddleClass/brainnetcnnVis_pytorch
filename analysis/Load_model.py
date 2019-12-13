model = BrainNetCNN(torch.nn.Module)
model.load_state_dict(torch.load('models/BNCNN12-10-01-25_model.pt'))
model.eval()