import torch
print("cuda avail:", torch.cuda.is_available())
print("cuda ver:", torch.version.cuda)
print("dev:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
