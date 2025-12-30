# Running Experiments on Windows (No Admin Required)

This guide explains how to run the experiment scripts on Windows without administrator permissions.

## Option 1: Python Scripts (Recommended - No Admin Needed)

The easiest way is to use the Python versions of the scripts, which work on Windows, Linux, and macOS without any additional setup.

### Prerequisites
- Python 3.6+ installed
- All project dependencies installed (PyTorch, Hydra, etc.)

### Usage

**For CIFAR-100 experiments:**
```cmd
python run_all_cifar.py
```

**For ImageNet-100 experiments:**
```cmd
python run_all_imagenet.py
```

### Advantages
- ✅ Works on Windows, Linux, and macOS
- ✅ No admin permissions needed
- ✅ No additional software installation required
- ✅ Same functionality as bash scripts

---

## Option 2: Git Bash (No Admin Needed)

If you have Git for Windows installed, you can use Git Bash to run the `.sh` scripts.

### Prerequisites
- Git for Windows installed (download from https://git-scm.com/download/win)
  - Git Bash is included automatically

### Usage

1. Open **Git Bash** (right-click in folder → "Git Bash Here" or search for "Git Bash" in Start menu)

2. Navigate to the project directory:
```bash
cd /c/path/to/MoE-Adapters/cil
```

3. Run the scripts:
```bash
bash run_all_cifar.sh
# or
bash run_all_imagenet.sh
```

### Advantages
- ✅ No admin permissions needed
- ✅ Works with existing `.sh` scripts
- ✅ Git Bash is lightweight and commonly installed

---

## Option 3: PowerShell Scripts (Alternative)

If you prefer PowerShell, you can convert the bash scripts. However, the Python versions are recommended as they're cross-platform.

---

## Option 4: WSL (Windows Subsystem for Linux)

**Note:** WSL typically requires admin permissions for initial setup, so this may not be suitable if you don't have admin access.

If WSL is already installed on your system:

1. Open WSL terminal
2. Navigate to the project (Windows drives are mounted at `/mnt/c/`)
3. Run the bash scripts normally

---

## Troubleshooting

### Python Script Issues

**Issue:** `python` command not found
- **Solution:** Try `python3` instead, or use the full path to Python
- **Windows:** You may need to add Python to PATH during installation

**Issue:** Colors don't display correctly in Windows Command Prompt
- **Solution:** The scripts will still work, just without colors. For better color support:
  - Use Windows Terminal (available from Microsoft Store)
  - Use PowerShell instead of Command Prompt
  - Install `colorama`: `pip install colorama` (then modify scripts to use it)

**Issue:** `CUDA_VISIBLE_DEVICES` not working
- **Solution:** On Windows, CUDA device selection works the same way. Make sure CUDA is properly installed and PyTorch can detect your GPU.

### Git Bash Issues

**Issue:** Line ending errors (`\r` characters)
- **Solution:** The scripts should work, but if you encounter issues, you can convert line endings:
  ```bash
  dos2unix run_all_cifar.sh
  dos2unix run_all_imagenet.sh
  ```
  Or in Git Bash, set: `git config --global core.autocrlf false`

---

## Recommended Approach

**For Windows users without admin access:**
1. Use the **Python scripts** (`run_all_cifar.py` and `run_all_imagenet.py`)
2. They work identically to the bash versions
3. No additional setup required beyond having Python installed

**Example:**
```cmd
cd C:\path\to\MoE-Adapters\cil
python run_all_cifar.py
```

---

## Summary

| Method | Admin Needed? | Setup Required | Cross-Platform |
|--------|---------------|----------------|----------------|
| Python Scripts | ❌ No | Python only | ✅ Yes |
| Git Bash | ❌ No | Git for Windows | ⚠️ Windows only |
| PowerShell | ❌ No | None (built-in) | ⚠️ Windows only |
| WSL | ✅ Yes (initial) | WSL setup | ⚠️ Windows only |

**Best choice for Windows without admin: Python scripts** 🐍
