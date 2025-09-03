import os
import subprocess

# List all installed packages
installed_packages = subprocess.check_output(["pip", "freeze"]).decode("utf-8").splitlines()

# Uninstall each package
for package in installed_packages:
    os.system(f"pip uninstall -y {package}")
