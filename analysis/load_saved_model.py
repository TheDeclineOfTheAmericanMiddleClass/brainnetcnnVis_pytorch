import torch

def main():
    # Adjust to load desired model
    loadpath_model = input('Filepath of model.pt : ')

    model = torch.load(loadpath_model)
    model.eval()  # setting to inference mode

if __name__ == '__main__':
    main()
