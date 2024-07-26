import llvmlite.ir as ir
import llvmlite.binding as llvm
import random
import string
import lief
import subprocess
from collections import Counter

def run_lifter(binary_path: str):
    print("Running retdec-decompiler for {}".format(binary_path))
    p = subprocess.run(["retdec-decompiler", "-s", "-k", binary_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, error = p.stdout, p.stderr
    print(output.decode())
    print(error.decode())
    p = subprocess.run(["llvm-dis", binary_path + ".bc", "-o", binary_path + ".ll"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, error = p.stdout, p.stderr
    print(output.decode())
    print(error.decode())

def get_random_string(length: int):
    return ''.join(random.choice(string.ascii_letters) for i in range(length))

def get_symbol_table(binary_path: str):
    binary = lief.ELF.parse(binary_path)
    symbols = binary.symbols
    function_symbols = {}
    for symbol in symbols:
        if symbol.type == lief.ELF.SYMBOL_TYPES.FUNC:
            function_symbols[symbol.value] = symbol.name
    return function_symbols

def use_random_name_table(before_binary, after_binary_path):
    before_strip_binary_path = before_binary
    before_strip_binary = lief.ELF.parse(before_strip_binary_path)
    before_strip_binary_symbols = before_strip_binary.symbols
    after_strip_binary = lief.ELF.parse(after_binary_path)


    function_symbols = []
    for symbol in before_strip_binary_symbols:
        if symbol.type == lief._lief.ELF.SYMBOL_TYPES.FUNC:
            function_symbols.append(symbol)

    name_to_random_name = {}
    random_name_to_name = {}

    for symbol in before_strip_binary_symbols:
        origin_name = symbol.name
        
        if "@GLIBC" in origin_name:
            continue
        
        new_symbol = before_strip_binary.get_symbol(symbol.name)
        new_symbol.name = get_random_string(10)
        after_strip_binary.add_static_symbol(new_symbol)
        random_name_to_name[new_symbol.name] = origin_name
        name_to_random_name[origin_name] = new_symbol.name

    after_strip_binary.write(after_binary_path)
    return name_to_random_name, random_name_to_name

def check_symbol_table(symbol_table: dict):
    # address: name
    duplicate_dict = {}
    
    name_count = Counter(symbol_table.values())
    for element, count in name_count.items():
        if count > 1:
            duplicate_dict[element] = 0
            
    for address in symbol_table:
        name = symbol_table[address]
        if name in duplicate_dict:
            if not duplicate_dict[name] == 0:
                symbol_table[address] = name+".{}".format(duplicate_dict[name])
            duplicate_dict[name] += 1
    # print(duplicate_dict)
    return symbol_table



def put_all_llvmlite(strip_ir_path: str, symbol_table: dict):
    function_symbols = check_symbol_table(symbol_table)
    with open(strip_ir_path, 'r') as f:
        llvm_assembly = f.read()
        f.close()
    
    llvm_module = llvm.parse_assembly(llvm_assembly)
    
    recovered = 0
    total = 0
    for f in llvm_module.functions:
        total += 1
        function_name = f.name
        if function_name.startswith("function_"):
            address = int(function_name.split("_")[1], 16)
            if address in function_symbols:
                origin_name = function_symbols[address]
                f.name = origin_name
                recovered += 1
        else:
            recovered += 1
    write_path = strip_ir_path[:-3] + ".llvmlite.ll"
    with open(write_path, 'w') as f:
        f.write(str(llvm_module))
        f.close()
        
    return write_path, total, recovered


def put_all_replace(strip_ir_path: str, symbol_table: dict):
    function_symbols = check_symbol_table(symbol_table)
    with open(strip_ir_path, 'r') as f:
        llvm_assembly = f.read()
        f.close()
        
    total = 0
    missing = 0
    plt = 0
    
    for function_symbol_address in function_symbols:
        if function_symbol_address == 0:
            plt += 1
            continue
        
        if "@GLIBC" in function_symbols[function_symbol_address]:
            plt += 1
            continue
        
        total += 1
        
        address = function_symbol_address
        
        function_name = "function_" + hex(address)[2:]
        
        if function_name not in llvm_assembly and function_symbols[function_symbol_address] not in llvm_assembly:
            missing += 1
            continue
        else:
            origin_name = function_symbols[function_symbol_address]
            llvm_assembly = llvm_assembly.replace(function_name, origin_name)
            
    write_path = strip_ir_path[:-3] + ".replace.ll"
    with open(write_path, 'w') as f:
        f.write(llvm_assembly)
        f.close()
        
    return write_path, total, total-missing, plt


def generate_recovered_ir(origin_binary_path: str, strip_ir_path: str, use_llvmlite=True):
    symbol_table = get_symbol_table(origin_binary_path)
    if use_llvmlite:
        llvmlite_ir, total, success = put_all_llvmlite(strip_ir_path, symbol_table)
        return llvmlite_ir, total, success
    else:
        replace_ir, total, success, plt = put_all_replace(strip_ir_path, symbol_table)
        return replace_ir, total, success
    
    
if __name__ == '__main__':
    origin_binary_path = "/home/username/symbol_test/arm-32_putty-0.74-O0_pscp"
    strip_ir_path = "/home/username/symbol_test/arm-32_putty-0.74-O0_pscp.strip.ll"
    generate_recovered_ir(origin_binary_path, strip_ir_path)
    print("Done")