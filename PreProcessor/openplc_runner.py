from DataGenerator import DataGenerator, DataGeneratorMultiProcessing
from FileScanner import FileScanner
from GraphConverter import Converter
from dataclasses import dataclass
from line_profiler import LineProfiler
import argparse
import pickle
import yaml
import os
import re

@dataclass
class Config:
    read_op: bool = False
    read_cache: bool = False
    max_length: int = 500
    save_path: str = 'coreutil_dataset'
    op_file: str = 'coreutil_dataset/op_file.pkl'
    root_path: str = 'coreutil_dataset'
    k_fold: int = 0
    

def parse_yaml(config_file) -> dict:
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rebuild_all', action='store_true', help='whether to rebuild data cache and op file')
    parser.add_argument('--rebuild_cache', action='store_true', help='whether to rebuild data cache')
    parser.add_argument('--rebuild_op', action='store_true', help='whether to rebuild op file')
    parser.add_argument('--read_op', action='store_true', help="whether to read op_file.pkl")
    parser.add_argument('--read_cache', action='store_true', help='whether to read data cache')
    parser.add_argument('--max_length', type=int, default=500, help="Max length of adj matrix")
    parser.add_argument('--save_path', type=str, default='coreutil_dataset', help="Path to save data, it should be a directory")
    parser.add_argument('--op_file', type=str, default='coreutil_dataset/op_file.pkl', help='Path to save op_file.pkl')
    parser.add_argument('--root_path', type=str, default='coreutil_dataset', help='Path to root directory of orginal dataset, only for file scan')
    parser.add_argument('--k_fold', type=int, default=0, help='set 0 for no k_fold, set k for k_fold')
    parser.add_argument('--config', "-c", type=str, default='./data_config.yaml', help='The Configuration file, if with this, no need to give other')
    return parser.parse_args()

def read_config() -> Config:
    config = Config()
    args = parse_arg()
    if args.config:
        yaml_config = parse_yaml(args.config)
        config.read_op = yaml_config['cache']['read_op']
        config.read_cache = yaml_config['cache']['read_cache']
        config.max_length = yaml_config['max_length']
        config.op_file = yaml_config['file_path']['op_path']
        config.root_path = yaml_config['file_path']['root_path']
        config.save_path = yaml_config['file_path']['save_path']
        config.k_fold = yaml_config['k_fold']
    else:
        config.read_op = args.read_op
        config.read_cache = args.read_cache
        config.max_length = args.max_length
        config.op_file = args.op_file
        config.root_path = args.root_path
        config.save_path = args.save_path
        config.k_fold = args.k_fold
        
    if args.rebuild_all:
        config.read_cache = False
        config.read_op = False
    
    if args.rebuild_cache:
        config.read_cache = False
    
    if args.rebuild_op:
        config.read_op = False
        
    return config


class OpenPLCScanner(FileScanner):
    def __init__(self, root_path: str):
        super().__init__(root_path)
        
    def scan(self):
        for folder_name in os.listdir(self.root_path):

            # Remove irrelevant directory
            if '-' not in folder_name:
                continue
            
            # If it is not a directory, skip
            if not os.path.isdir(os.path.join(self.root_path, folder_name)):
                continue

            if "arm" in folder_name:
                arch = "arm"
            elif "mips" in folder_name:
                arch = "mips"
            elif "powerpc" in folder_name:
                arch = "powerpc"
            else:
                arch = "x86"
                
            if "O0" in folder_name:
                opt_level = "O0"
            elif "O1" in folder_name:
                opt_level = "O1"
            elif "O2" in folder_name:
                opt_level = "O2"
            elif "O3" in folder_name:
                opt_level = "O3"
            else:
                opt_level = "O0"

            binary_name = 'openplc'
            dot_file_path = os.path.join(self.root_path, folder_name)
            
            self.file_tree.add_to_tree(binary_name=binary_name, dot_file=dot_file_path, arch=arch, opt=opt_level)
            
        return self.file_tree

def main():
    config: Config = read_config()
    file_tree = OpenPLCScanner(root_path=config.root_path).scan()
    # file_tree = FileScanner(root_path=config.root_path).scan()
    converter = Converter(op_file=config.op_file, read_op=config.read_op, max_length=config.max_length)
    # data_generator = DataGenerator(file_tree=file_tree, save_path=config.save_path, converter=converter, read_cache=config.read_cache)
    # data_generator.run()
    
    multi_datagen = DataGeneratorMultiProcessing(file_tree=file_tree, save_path=config.save_path, converter=converter, read_cache=config.read_cache)
    multi_datagen.run()
    
if __name__ == "__main__":
    main()
