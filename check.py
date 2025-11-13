import torch

for path in ["bestobj.torchscript.pt", "pathseg.torchscript.pt"]:
    try:
        model = torch.jit.load(path)
        print(f"{path} loaded successfully ✅")
        x = torch.randn(1, 3, 640, 640)
        y = model(x)
        if torch.is_tensor(y):
            print(f"{path} output tensor shape: {tuple(y.shape)}")
        else:
            print(f"{path} output type: {type(y)}")
    except Exception as e:
        print(f"Error loading {path}: {e}")
