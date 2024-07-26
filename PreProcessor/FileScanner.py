import os
from typing import Dict


# Target:
# Dataset -> Binary -> Function -> different function list and indicate the opt/arch


# FileTree:
# Data -> Binary -> Arch -> Opt -> Directory
class FileTree:
    def __init__(self):
        self.pool: Dict = {}

    def add_to_tree(self, binary_name: str, dot_file: str, arch: str, opt: str):

        if opt not in self.pool:
            self.pool[opt] = {}

        if arch not in self.pool[opt]:
            self.pool[opt][arch] = {}

        if binary_name not in self.pool[opt][arch]:
            self.pool[opt][arch][binary_name] = {}

        self.pool[opt][arch][binary_name] = dot_file

    def __iter__(self):
        for opt in self.pool:
            for arch in self.pool[opt]:
                for binary_name in self.pool[opt][arch]:
                    yield self.pool[opt][arch][binary_name], opt, arch, binary_name

    def __len__(self):
        count = 0
        for opt in self.pool:
            for arch in self.pool[opt]:
                for binary_name in self.pool[opt][arch]:
                    count += len(os.listdir(self.pool[opt][arch][binary_name]))
        return count


class FileScanner:
    def __init__(self, root_path: str):
        self.root_path = root_path
        self.file_tree = FileTree()

    def scan(self):
        for arch_bin in os.listdir(self.root_path):

            # Remove irrelevant directory
            if "-" not in arch_bin:
                continue

            arch = arch_bin[:-3]
            opt_level = arch_bin[-3:]

            for binary_path in os.listdir(
                os.path.join(self.root_path, arch_bin, "lift")
            ):
                binary_now_path = os.path.join(
                    self.root_path, arch_bin, "lift", binary_path
                )

                # Remove irrelevant
                if not (
                    os.path.isdir(binary_now_path) and binary_path.startswith("c_dot_")
                ):
                    continue

                binary_name = binary_path.replace("c_dot_", "")
                self.file_tree.add_to_tree(
                    binary_name=binary_name,
                    dot_file=binary_now_path,
                    arch=arch,
                    opt=opt_level,
                )
        return self.file_tree
