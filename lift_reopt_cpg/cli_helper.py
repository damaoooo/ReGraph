import psutil
import enum
import subprocess
import os
import shutil

# Because of multiprocessing, I cannot write OOP code, because the class object cannot be pickled.
# Therefore, I have to write the functions in a procedural way. And write a big class to wrap all the functions.

JOERN_PATH = "/home/damaoooo/Downloads/joern-cli"

# Get total system memory in bytes
total_memory = psutil.virtual_memory().total

# Convert bytes to Gigabytes (optional)
memory_in_gb = total_memory / (1024**3)


class unit(enum.Enum):
    B = 1
    KB = 1024
    MB = 1024 * 1024
    GB = 1024 * 1024 * 1024


def lift_reoptimize(file_path: str, max_memory: int = 5 * unit.GB.value, timeout: int = 3600):
    """
    Reoptimizes a given binary file using RetDec and LLVM tools.
    This function performs the following steps:
    1. Changes the current working directory to the directory containing the file.
    2. Runs RetDec decompiler on the file with specified memory and timeout constraints.
    3. Converts the decompiled output to LLVM IR using llvm-dis.
    4. Compiles the LLVM IR to a reoptimized binary using clang.
    5. Runs RetDec decompiler on the reoptimized binary.
    Args:
        file_path (str): The path to the binary file to be reoptimized.
        max_memory (int, optional): The maximum memory to be used by RetDec in bytes. Defaults to 5 GB.
        timeout (int, optional): The maximum time allowed for each subprocess to run in seconds. Defaults to 3600 seconds.
    Returns:
        bool: True if the reoptimization process completes successfully, False if a timeout occurs.
    """

    success = True
    dir_path = os.path.dirname(file_path)
    os.chdir(dir_path)
    file_name = os.path.basename(file_path)
    try:
        m = subprocess.run(["retdec-decompiler", file_name, "-s", "--max-memory", str(max_memory), "-k", "--backend-keep-library-funcs"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
        m = subprocess.run(["llvm-dis", "{}.bc".format(file_name), "-o", "{}.ll".format(file_name)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
        m = subprocess.run(["clang", "-m32", "-O3", "-c", "{}.ll".format(file_name), "-fno-inline-functions", "-o", "{}.re".format(file_name)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
        m = subprocess.run(["retdec-decompiler", "{}.re".format(file_name), "-s", "--max-memory", str(max_memory), "-k", "--backend-keep-library-funcs"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
    except subprocess.TimeoutExpired:
        success = False
    return success

def convert_to_cpg(file_path: str, memory_size: int = 5, timeout: int = 3600):
    success = True
    dir_path = os.path.dirname(file_path)
    os.chdir(dir_path)
    file_name = os.path.basename(file_path)
    joern_dir = JOERN_PATH
    c2cpg_path = os.path.join(joern_dir, "c2cpg.sh")
    try:
        subprocess.run([c2cpg_path, "-J-Xmx{}G".format(memory_size), "{}.re.c".format(file_name), "-o", "{}.cpg".format(file_name)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)

    except subprocess.TimeoutExpired:
        success = False
    return success

def export_cpg(file_path: str, memory_size: int = 5, timeout: int = 3600):
    success = True
    dir_path = os.path.dirname(file_path)
    os.chdir(dir_path)
    file_name = os.path.basename(file_path)
    joern_dir = JOERN_PATH
    current_env = os.environ.copy()
    current_env["JAVA_OPTS"] = "-Xmx{}g".format(memory_size)
    joern_export_path = os.path.join(joern_dir, "joern-export")
    try:
        subprocess.run([joern_export_path, "{}.cpg".format(file_name), "-o", "c_dot_{}".format(file_name)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout, env=current_env)
    except subprocess.TimeoutExpired:
        success = False
    return success

def clean_up_single(file_path: str):
    suffix_list = [".dsm", ".config.json", ".bc", ".c", ".ll", ".cpg", ".re"]
    for suffix in suffix_list:
        for file in os.listdir(file_path):
            if file.endswith(suffix):
                os.remove(os.path.join(file_path, file))
            if file.startswith("workspace") or file.startswith("out"):
                shutil.rmtree(os.path.join(file_path, file))


def clean_up_single(file_path: str, clean_all: bool = False, lift_opt_only: bool = False):
    suffix_list = [".dsm", ".config.json", ".bc", ".c", ".ll", ".cpg", ".re"]
    if lift_opt_only:
        # Keep .re
        suffix_list.remove(".re")

    for suffix in suffix_list:
        for file in os.listdir(file_path):
            if file.endswith(suffix):
                os.remove(os.path.join(file_path, file))
            if file.startswith("workspace") or file.startswith("out"):
                shutil.rmtree(os.path.join(file_path, file))
            # if it is a folder start with "c_dot", remove it
            if file.startswith("c_dot") and clean_all:
                shutil.rmtree(os.path.join(file_path, file))