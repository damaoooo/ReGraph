import subprocess
from tqdm import tqdm
import os
import time
import shutil
import multiprocessing
import pandas as pd
import enum
import re
import psutil
from nb_log import get_logger
from llvm_name_recover import generate_recovered_ir

logger = get_logger("lift_process_strip", log_filename="lift_process_strip.log", log_path="/home/username/plc_test/BSC_reproduce", is_add_stream_handler=False)
# logging.basicConfig(filename="lift_process_strip.log", level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", encoding='utf-8', filemode='w')

# Get total system memory in bytes
total_memory = psutil.virtual_memory().total

# Convert bytes to Gigabytes (optional)
memory_in_gb = total_memory / (1024 ** 3)

"""
    retdec-decompiler ${filename} -s -k --backend-keep-library-funcs --max-memory 182536110080
    llvm-dis ${filename}.bc -o ${filename}.ll
    clang -m32 -O3 -c ${filename}.ll -fno-inline-functions -o ${filename}.re
    retdec-decompiler ${filename}.re -s -k --backend-keep-library-funcs --max-memory 182536110080
    JAVA_OPTS='-Xmx128g' ${joern_dir}/c2cpg.sh -J-Xmx128g ${filename}.re.c -o ${filename}.cpg
    JAVA_OPTS='-Xmx128g' ${joern_dir}/joern-export ${filename}.cpg -o c_dot_${filename}
    
function clean_dir() {
    this_path=$1
    rm -rf ${this_path}*.dsm
    rm -rf ${this_path}*.config.json
    rm -rf ${this_path}*.bc
    rm -rf ${this_path}*.c
    rm -rf ${this_path}*.ll
    rm -rf ${this_path}*.cpg
    rm -rf ${this_path}*.re
}
    
"""

class unit(enum.Enum):
    B = 1
    KB = 1024
    MB = 1024 * 1024
    GB = 1024 * 1024 * 1024


def extract_file_info_from_idb(idb_path: str):
    # idb_path: IDBs/Dataset-1/z3/arm32-clang-5.0-O0_z3.i64
    # return project file "z3" and file_name "arm32-clang-5.0-O0_z3"
    
    project_file = os.path.dirname(idb_path)
    project_name = os.path.basename(project_file)
    file_name = os.path.basename(idb_path).replace(".i64", "")
    return project_name, file_name

def read_file_function(csv_path: str, file_path: str):
    result = []
    csv_file = pd.read_csv(csv_path)
    groups = csv_file.groupby(["idb_path"]).groups.items()
    for group_name, content in groups:
        func_list = csv_file.loc[content, "func_name"].to_list()
        project_name, file_name = extract_file_info_from_idb(group_name)
        if "Dataset" in project_name:
            project_name = ""
        corresponding_file_path = os.path.join(file_path, project_name, file_name)
        result.append([corresponding_file_path, func_list])
    return result

def lift_reoptimize(file_path: str, origin_file_path: str, func_list: list[str] = None, max_memory: int = 5 * unit.GB.value, timeout: int = 3600):
    success = True
    dir_path = os.path.dirname(file_path)
    os.chdir(dir_path)
    file_name = os.path.basename(file_path)
    try:
        if func_list is not None:
            m = subprocess.run(["retdec-decompiler", file_name, "-s", "--select-functions", ','.join(func_list), "--max-memory", str(max_memory)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
            m = subprocess.run(["llvm-dis", "{}.bc".format(file_name), "-o", "{}.ll".format(file_name)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
            m = subprocess.run(["clang", "-m32", "-O3", "-c", "{}.ll".format(file_name), "-fno-inline-functions", "-o", "{}.re".format(file_name)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
            m = subprocess.run(["retdec-decompiler", "{}.re".format(file_name), "-s", "--select-functions", ','.join(func_list), "--max-memory", str(max_memory)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
        else:
            m = subprocess.run(["retdec-decompiler", file_name, "-s", "--max-memory", str(max_memory), "-k", "--backend-keep-library-funcs"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
            m = subprocess.run(["llvm-dis", "{}.bc".format(file_name), "-o", "{}.ll".format(file_name)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
            new_ir, total, success = generate_recovered_ir(origin_file_path, "{}.ll".format(file_name), use_llvmlite=False)
            logger.debug("not using LLVMlite recover, file: {} success: {}, total: {}".format(file_path, success, total))
            m = subprocess.run(["clang", "-m32", "-O3", "-c", new_ir, "-fno-inline-functions", "-o", "{}.re".format(file_name)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
            if m.returncode != 0:
                logger.warning("lift optimize, clang failed for file: {}".format(file_path))
                success = False
            m = subprocess.run(["retdec-decompiler", "{}.re".format(file_name), "-s", "--max-memory", str(max_memory), "-k", "--backend-keep-library-funcs"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
    except subprocess.TimeoutExpired:
        logger.warning("lift optimize, Timeout for file: {}".format(file_path))
        success = False
    except Exception as e:
        logger.error("lift optimize, Exception for file: {}, error: {}".format(file_path, e))
        success = False
    except:
        logger.error("lift optimize, Unknown error for file: {}".format(file_path))
        success = False
    return success

def convert_to_cpg(file_path: str, memory_size: int, timeout: int = 3600):
    success = True
    dir_path = os.path.dirname(file_path)
    os.chdir(dir_path)
    file_name = os.path.basename(file_path)
    joern_dir = "/home/username/Downloads/joern-cli"
    try:
        subprocess.run(["{}/c2cpg.sh".format(joern_dir), "-J-Xmx{}G".format(memory_size), "{}.re.c".format(file_name), "-o", "{}.cpg".format(file_name)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
    
    except subprocess.TimeoutExpired:
        success = False
        logger.warn("convert to CPG, Timeout for file: {}".format(file_path))
    return success

def export_cpg(file_path: str, memory_size: int, timeout: int = 3600):
    success = True
    dir_path = os.path.dirname(file_path)
    os.chdir(dir_path)
    file_name = os.path.basename(file_path)
    joern_dir = "/home/username/Downloads/joern-cli"
    current_env = os.environ.copy()
    current_env["JAVA_OPTS"] = "-Xmx{}g".format(memory_size)
    try:
        subprocess.run(["{}/joern-export".format(joern_dir), "{}.cpg".format(file_name), "-o", "c_dot_{}".format(file_name)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout, env=current_env)
    except subprocess.TimeoutExpired:
        success = False
        logger.warn("export CPG, Timeout for file: {}".format(file_path))
    return success

def clean_up_single(file_path: str):
    suffix_list = [".dsm", ".config.json", ".bc", ".c", ".ll", ".cpg", ".re"]
    for suffix in suffix_list:
        for file in os.listdir(file_path):
            if file.endswith(suffix):
                os.remove(os.path.join(file_path, file))
            if file.startswith("workspace") or file.startswith("out"):
                shutil.rmtree(os.path.join(file_path, file))


def file_iterator_dataset_1(root_path: str):
    result = []
    for project in os.listdir(root_path):
        project_path = os.path.join(root_path, project)
        
        if not os.path.isdir(project_path):
            continue
        
        for file in os.listdir(project_path):
            file_path = os.path.join(project_path, file)
            result.append(file_path)
            
    return result

def file_iterator_dataset_2(root_path: str):
    result = []
    ends_list = [".i64", ".idb", ".c", ".asm", ".bc", ".ll", ".re", ".cpg", ".dsm", ".config.json"]
    starts_list = ['arm-32', 'mips-32', 'x86-32']
    
    def is_end_with(file_name: str):
        for end in ends_list:
            if file_name.endswith(end):
                return True
        return False
    
    def is_start_with(file_name: str):
        for start in starts_list:
            if file_name.startswith(start):
                return True
        return False
    
    for file in os.listdir(root_path):
        if is_end_with(file):
            continue
        
        if not is_start_with(file):
            continue
        
        if os.path.isdir(os.path.join(root_path, file)):
            continue

        
        result.append(os.path.join(root_path, file))
        
    return result

def extract_file_info(file_path: str):
    file_name = os.path.basename(file_path)
    project_name = os.path.basename(os.path.dirname(file_path))
    return project_name, file_name


def runner(csv_path: str, file_path: str, empty_csv_path: str = "", is_dataset2: bool = False):
    
    # Print a banner for the steps
    # print("Step 1: Lifting and Reoptimizing")
    # print("Step 2: Generate CPG")
    # print("Step 3: Exporting CPG")
    # print("Step 4: Cleaning Up")
    
    print("Step 1: Read file list and function list")
    print("Step 2: Lifting and Reoptimizing")
    print("Step 3: Generate CPG")
    print("Step 4: Exporting CPG")
    print("Step 5: Cleaning Up")
    
    if is_dataset2:
        iter_list = file_iterator_dataset_2(file_path)
    else:
        iter_list = read_file_function(csv_path, file_path)
        
        if empty_path:
            iter_list = empty_intersection(iter_list, file_path, empty_csv_path)
    
    success_lift = []
    
    
    
    if empty_path:
        # find all the memory
        memory_size = round(memory_in_gb) - 2
        min_memory = 64
        processor_num = memory_size // min_memory
        cpu_count = max(1, processor_num)
    else:
        cpu_count = multiprocessing.cpu_count()
        min_memory = 5
    
    if empty_path:
        lift_mem = min_memory * unit.GB.value
        joern_mem = min_memory
    else:
        lift_mem = min_memory * unit.GB.value
        joern_mem = min_memory
    
    # Step 1: Lift and Reoptimize, use multiprocessing to speed up
    bar1 = tqdm(total=len(iter_list), desc="Lifting and Reoptimizing", dynamic_ncols=True)
    pool = multiprocessing.Pool(processes=cpu_count)
    result_list = []
    for file_func_list in iter_list:
        if is_dataset2:
            file = file_func_list
            project_name, file_name = extract_file_info(file_func_list)
            origin_file = file.replace("/Dataset-2-strip/", "/Dataset-2/")
            
            if not os.path.exists(origin_file):
                logger.warn("runner: file not exists: {}".format(origin_file))
                bar1.update(1)
                continue
            
            p = pool.apply_async(lift_reoptimize, (file_func_list, origin_file, None, lift_mem, ), callback=lambda _: bar1.update(1))
        else:
            # Replace each function from function lift, sub_1234 -> function_1234
            file, func_list = file_func_list
            for i in range(len(func_list)):
                if not isinstance(func_list[i], str):
                    continue
                if re.search(r"sub_[0-9a-fA-F]+", func_list[i]):
                    func_list[i] = func_list[i].replace("sub_", "function_")
            
            func_list = [x for x in func_list if isinstance(x, str)]
            
            if not func_list:
                continue
            
            project_name, file_name = extract_file_info(file)
            p = pool.apply_async(lift_reoptimize, (file, func_list, lift_mem, ), callback=lambda _: bar1.update(1))
        result_list.append([p, file])
    pool.close()
    pool.join()
    for p, file in result_list:
        if p.get():
            success_lift.append(file)
    bar1.close()
    print("Step 1: Lifting and Reoptimizing - Success Rate({}): {}/{}".format(round(len(success_lift) / len(iter_list)), len(success_lift), len(iter_list)))

    
    # Step 2: Export C into CPG file in Parallel
    success_convert = []
    pool2 = multiprocessing.Pool(processes=cpu_count//2)
    result_list = []
    bar2 = tqdm(total=len(success_lift), desc="Converting CPG", dynamic_ncols=True)
    for file in success_lift:
        project_name, file_name = extract_file_info(file)
        p = pool2.apply_async(convert_to_cpg, (file, joern_mem), callback=lambda _: bar2.update(1))
        result_list.append([p, file])

    pool2.close()
    pool2.join()
    for p, file in result_list:
        if p.get():
            success_convert.append(file)
    bar2.close()
    print("Step 2: Generate CPG - Success Rate({}): {}/{}".format(round(len(success_convert) / len(success_lift)), len(success_convert), len(success_lift)))
    
    
    # Step 3: Export the CPG into c_dot_ files
    success_export = []
    pool3 = multiprocessing.Pool(processes=cpu_count//2)
    result_list = []
    bar3 = tqdm(total=len(success_convert), desc="Exporting CPG", dynamic_ncols=True)
    for file in success_convert:
        project_name, file_name = extract_file_info(file)
        p = pool3.apply_async(export_cpg, (file, joern_mem), callback=lambda _: bar3.update(1))
        result_list.append([p, file])
        
    pool3.close()
    pool3.join()
    for p, file in result_list:
        if p.get():
            success_export.append(file)
    bar3.close()
    print("Step 3: Exporting CPG - Success Rate({}): {}/{}".format(round(len(success_export) / len(success_convert)), len(success_export), len(success_convert)))
    
    success_clean = []
    bar4 = tqdm(total=len(success_convert), desc="Cleaning Up", dynamic_ncols=True)
    for file in success_export:
        file = file.replace(".cpg", "")
        project_name, file_name = extract_file_info(file)
        bar4.set_postfix_str("{}/{}".format(project_name, file_name))
        clean_up_single(os.path.dirname(file))
        success_clean.append(file)
        bar4.update(1)
        
    bar4.close()
    
    print("Step 4: Cleaning Up - Success Rate({}): {}/{}".format(round(len(success_clean) / len(success_convert)), len(success_clean), len(success_convert)))
    
def empty_intersection(iter_list: str, file_path: str, empty_csv_path: str):
    empty_csv = pd.read_csv(empty_csv_path)
    """
    empty_csv: arch,project,opt_level,file_name,c_dot
    we iter the c_dot column and find the corresponding file in the file_path
    """
    
    # build a dict to speedup
    lookup_dict = {}
    for file, func_list in iter_list:
        lookup_dict[file] = func_list
    
    new_result = []
    
    for c_dot in empty_csv["c_dot"]:
        file_name = c_dot.replace("c_dot_", "")
        full_file_path = os.path.join(file_path, file_name)
        print("file_path:", full_file_path)
        new_result.append([full_file_path, lookup_dict[full_file_path]])
        
    return new_result


def clean_up(file_path: str, is_dataset2: bool = False):
    
    if is_dataset2:
        for file_name in os.listdir(file_path):
            if file_name.startswith("c_dot_out") or file_name.startswith("c_dot_workspace"):
                shutil.rmtree(os.path.join(file_path, file_name))
            
            if file_name.startswith("c_dot_"):
                shutil.rmtree(os.path.join(file_path, file_name))
                
            if os.path.isdir(os.path.join(file_path, file_name)):
                shutil.rmtree(os.path.join(file_path, file_name))
                
        suffix_list = [".dsm", ".config.json", ".bc", ".c", ".ll", ".cpg", ".re"]
            
        for suffix in suffix_list:
            for file in os.listdir(file_path):
                if file.endswith(suffix):
                    os.remove(os.path.join(file_path, file))
    else:
        for project in os.listdir(file_path):
            project_path = os.path.join(file_path, project)
            if not os.path.isdir(project_path):
                continue
            for c_dot_file in os.listdir(project_path):
                if c_dot_file.startswith("c_dot_out") or c_dot_file.startswith("c_dot_workspace"):
                    shutil.rmtree(os.path.join(project_path, c_dot_file))
                if not c_dot_file.startswith("c_dot_"):
                    continue
                if not os.path.isdir(os.path.join(project_path, c_dot_file)):
                    continue
                shutil.rmtree(os.path.join(project_path, c_dot_file))
            # suffix clean
            suffix_list = [".dsm", ".config.json", ".bc", ".c", ".ll", ".cpg", ".re"]
            for suffix in suffix_list:
                for file in os.listdir(project_path):
                    if file.endswith(suffix):
                        os.remove(os.path.join(project_path, file))

if __name__ == "__main__":
    file_path = "/home/username/binary_function_similarity/Binaries/Dataset-2-strip/"
    csv_path = "/home/username/binary_function_similarity/DBs/Dataset-2/features/flowchart_Dataset-2.csv"
    empty_path = "/home/username/plc_test/BSC_reproduce/empty.csv"
    empty_path = ""
    
    clean_up(file_path, is_dataset2=True)
    runner(csv_path, file_path, empty_path, is_dataset2=True)