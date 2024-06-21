import importlib.util
import os

def is_torch_installed():
    return importlib.util.find_spec("torch") is not None

def update_requirements():
    with open("requirements.txt", "r") as f:
        requirements = f.read().splitlines()
    
    torch_req = "torch"
    if not is_torch_installed() and torch_req not in requirements:
        requirements.append(torch_req)
        with open("requirements.txt", "w") as f:
            f.write("\n".join(requirements))
        print("Added PyTorch to requirements.txt")
    elif is_torch_installed():
        print("PyTorch is already installed. Not adding to requirements.")

if __name__ == "__main__":
    update_requirements()
