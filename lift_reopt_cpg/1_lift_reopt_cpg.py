import rich.progress
from cli_helper import lift_reoptimize, convert_to_cpg, export_cpg, clean_up_single
from typing import List
import multiprocessing
import rich
import os

def find_common_folders(file_list: List[str]):
    # Find the folders of the files, without repeating
    folders = set()
    for file in file_list:
        folders.add(os.path.dirname(file))        
    return list(folders)

class BinaryProcess:
    def __init__(self, file_list: List, jobs: int = 1, lift_reopt_only: bool = False):
        self.file_list = file_list
        if jobs == -1:
            jobs = multiprocessing.cpu_count()
        # Check for valid jobs
        if jobs < 1:
            raise ValueError("Invalid number of jobs")
        self.jobs = jobs

        self.lift_reopt_only = lift_reopt_only

    def process_files_multi_processing(self):
        # 1. Lifting binaries
        lift_results = []
        
        with rich.progress.Progress(
            "[progress.description]{task.description}",
            rich.progress.SpinnerColumn(),
            rich.progress.BarColumn(),
            rich.progress.TaskProgressColumn(),
            transient=False
        ) as progress:
            lift_task = progress.add_task("🚀 Lifting binaries...", total=len(self.file_list))
            
            pool = multiprocessing.Pool(processes=self.jobs)
            
            for file in self.file_list:
                lift_result = pool.apply_async(lift_reoptimize, 
                                       args=(file,),
                                       callback=lambda _: progress.advance(lift_task))
                lift_results.append(lift_result)

            pool.close()
            pool.join()
            # Ensure lift_results is ordered as self.file_list
            lift_results = [result.get() for result in lift_results]

            if not self.lift_reopt_only:
                # 2. Joern to parse lifted binaries
                pool = multiprocessing.Pool(processes=self.jobs)
                cpg_list = [x for x, y in zip(self.file_list, lift_results) if y]
                cpg_results = []
                lift_task = progress.add_task("🔍 Parsing lifted binaries...", total=len(cpg_list))
                for file in cpg_list:
                    cpg_result = pool.apply_async(convert_to_cpg, 
                                        args=(file,),
                                        callback=lambda _: progress.advance(lift_task))
                    cpg_results.append(cpg_result)
                    
                pool.close()
                pool.join()
                
                cpg_results = [result.get() for result in cpg_results]
                
                # 3. Exporting CPGs
                pool = multiprocessing.Pool(processes=self.jobs)
                export_list = [x for x, y in zip(self.file_list, cpg_results) if y]
                export_results = []
                lift_task = progress.add_task("📤 Exporting CPGs...", total=len(export_list))
                for file in export_list:
                    export_result = pool.apply_async(export_cpg, 
                                        args=(file,),
                                        callback=lambda _: progress.advance(lift_task))
                    export_results.append(export_result)
                
                pool.close()
                pool.join()
                
                export_results = [result.get() for result in export_results]
            
            # 4. Clean up
            pool = multiprocessing.Pool(processes=self.jobs)
            clean_list = find_common_folders(self.file_list)
            clean_task = progress.add_task("🧹 Cleaning up...", total=len(self.file_list))
            for folder in clean_list:
                pool.apply_async(clean_up_single, 
                                       args=(folder, False, self.lift_reopt_only),
                                       callback=lambda _: progress.advance(clean_task))
                
            pool.close()
            pool.join()
            
            
    def process_files_single_processing(self):
        # 1. Lifting binaries
        lift_results = []
        
        with rich.progress.Progress(
            "[progress.description]{task.description}",
            rich.progress.SpinnerColumn(),
            rich.progress.BarColumn(),
            rich.progress.TaskProgressColumn(),
            transient=False
        ) as progress:
            lift_task = progress.add_task("🚀 Lifting binaries...", total=len(self.file_list))
            
            for file in self.file_list:
                lift_result = lift_reoptimize(file)
                lift_results.append(lift_result)
                progress.advance(lift_task)

            # 2. Joern to parse lifted binaries
            cpg_list = [x for x, y in zip(self.file_list, lift_results) if y]
            print("Lifted binaries / Total binaries: ", len(cpg_list), "/", len(self.file_list))
            cpg_results = []
            lift_task = progress.add_task("🔍 Parsing lifted binaries...", total=len(cpg_list))
            for file in cpg_list:
                cpg_result = convert_to_cpg(file)
                cpg_results.append(cpg_result)
                progress.advance(lift_task)
            
            # 3. Exporting CPGs
            export_list = [x for x, y in zip(self.file_list, cpg_results) if y]
            export_results = []
            print("Parsed binaries / Total binaries: ", len(export_list), "/", len(self.file_list))
            lift_task = progress.add_task("📤 Exporting CPGs...", total=len(export_list))
            for file in export_list:
                export_result = export_cpg(file)
                export_results.append(export_result)
                progress.advance(lift_task)
                
            success = [x for x, y in zip(self.file_list, export_results) if y]
            print("Exported binaries / Total binaries: ", len(success), "/", len(self.file_list))
            
            # 4. Clean up
            clean_list = find_common_folders(self.file_list)
            print("Cleaning up...")
            for folder in clean_list:
                clean_up_single(folder, clean_all=False, lift_reopt_only=self.lift_reopt_only)

            return len(success)
        
    def process_files(self):
        if self.jobs == 1:
            return self.process_files_single_processing()
        else:
            return self.process_files_multi_processing()
        

if __name__ == "__main__":
    start_path = "/home/damaoooo/ReGraph/graph_dataset/binaries_openplc"
    file_list = []
    
    for arch in os.listdir(start_path):
        arch_path = os.path.join(start_path, arch)
        for opt in os.listdir(arch_path):
            opt_path = os.path.join(arch_path, opt)
            for file in os.listdir(opt_path):
                file_list.append(os.path.join(opt_path, file))

    # Clean up first
    file_clean_list = find_common_folders(file_list)
    for folder in file_clean_list:
        clean_up_single(folder, clean_all=True)


    file_list = []

    for arch in os.listdir(start_path):
        arch_path = os.path.join(start_path, arch)
        for opt in os.listdir(arch_path):
            opt_path = os.path.join(arch_path, opt)
            for file in os.listdir(opt_path):
                file_list.append(os.path.join(opt_path, file))
                
    # print("Total binaries: ", len(file_list))

    binary_process = BinaryProcess(file_list, jobs=-1)
    binary_process.process_files()