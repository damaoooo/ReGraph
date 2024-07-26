import subprocess
import os
from glob import glob
import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Generate the comparison result")
    parser.add_argument(
        "--ida_path",
        type=str,
        default="/home/sentry2/idapro-8.4",
        help="The path of IDA Pro",
    )
    parser.add_argument(
        "-d",
        "--data_path",
        type=str,
        default="dataset/openplc_compare",
        help="The path of all files",
    )
    return parser.parse_args()


args = parse_args()

ida_path = args.ida_path


def generate_idb(file_path):
    subprocess.run([f"{ida_path}/idat64", "-B", f"{file_path}"])


def generate_bin_export(file_dir: str):
    subprocess.run(["bindiff", "--export", f"{file_dir}"])


def generate_bin_diff(filename_1: str, filename_2: str):
    subprocess.run(
        [
            "bindiff",
            "--primary",
            filename_1 + ".BinExport",
            "--secondary",
            filename_2 + ".BinExport",
        ]
    )


def clear_files(full_path: str):
    os.remove(full_path + ".i64")
    os.remove(full_path + ".asm")


def remove_ext(folder_path: str, ext: str):
    # Remove all files with the given extension like: a.i64, a.asm, a.BinExport
    for file in glob(folder_path + "/*." + ext):
        os.remove(file)


def generate_essential_files(all_path):
    remove_ext(all_path, "i64")
    remove_ext(all_path, "asm")
    remove_ext(all_path, "BinExport")
    remove_ext(all_path, "BinDiff")
    for file in os.listdir(all_path):
        full_path = os.path.join(all_path, file)
        if os.path.isdir(full_path):
            continue

        if not os.path.isfile(full_path):
            continue

        generate_idb(full_path)
    remove_ext(all_path, "asm")
    generate_bin_export(all_path)


def generate_compare_database(all_path):
    remove_ext(all_path, "BinDiff")
    arch_list = [
        "g++",
        "arm-linux-gnueabi-g++",
        "mips-linux-gnu-g++",
        "powerpc-linux-gnu-g++",
    ]

    abs_all_path = os.path.abspath(all_path)
    current_path = os.getcwd()
    os.chdir(abs_all_path)
    for arch1 in arch_list:
        for arch2 in arch_list:
            arch1_O0 = f"Res0_{arch1}-O0.o"
            arch2_O3 = f"Res0_{arch2}-O3.o"
            arch1_O0_path = os.path.join(abs_all_path, arch1_O0)
            arch2_O3_path = os.path.join(abs_all_path, arch2_O3)
            generate_bin_diff(arch1_O0_path, arch2_O3_path)

            arch1_O0_re = f"Res0_{arch1}-O0.o_re"
            arch2_O3_re = f"Res0_{arch2}-O3.o_re"
            arch1_O0_re_path = os.path.join(abs_all_path, arch1_O0_re)
            arch2_O3_re_path = os.path.join(abs_all_path, arch2_O3_re)
            generate_bin_diff(arch1_O0_re_path, arch2_O3_re_path)
    os.chdir(current_path)


def read_one_result_file(result_path):
    conn = sqlite3.connect(result_path)
    c = conn.cursor()
    similarity_score = c.execute(
        "select AVG(similarity) from function where name1=name2 and (name1 like '%_body_%' or name1 like '%_init_%');"
    ).fetchall()[0][0]
    conn.close()
    return similarity_score


def read_all_result_files(all_path):

    data = {
        ("ARM", "Before"): [0] * 5,
        ("ARM", "After"): [0] * 5,
        ("ARM", "Inc"): [0] * 5,
        ("PowerPC", "Before"): [0] * 5,
        ("PowerPC", "After"): [0] * 5,
        ("PowerPC", "Inc"): [0] * 5,
        ("MIPS", "Before"): [0] * 5,
        ("MIPS", "After"): [0] * 5,
        ("MIPS", "Inc"): [0] * 5,
        ("X86", "Before"): [0] * 5,
        ("X86", "After"): [0] * 5,
        ("X86", "Inc"): [0] * 5,
        ("AVG", "Before"): [0] * 5,
        ("AVG", "After"): [0] * 5,
        ("AVG", "Inc"): [0] * 5,
    }

    arch_list = [
        "g++",
        "arm-linux-gnueabi-g++",
        "mips-linux-gnu-g++",
        "powerpc-linux-gnu-g++",
    ]
    arch_mapping = {
        "g++": "X86",
        "arm-linux-gnueabi-g++": "ARM",
        "mips-linux-gnu-g++": "MIPS",
        "powerpc-linux-gnu-g++": "PowerPC",
    }
    index = ["ARM", "PowerPC", "MIPS", "X86", "AVG"]
    abs_all_path = os.path.abspath(all_path)
    for arch1 in arch_list:
        for arch2 in arch_list:
            arch1_O0 = f"Res0_{arch1}-O0.o"
            arch2_O3 = f"Res0_{arch2}-O3.o"

            arch1_O0_re = f"Res0_{arch1}-O0.o_re"
            arch2_O3_re = f"Res0_{arch2}-O3.o_re"

            before_bindiff = f"{arch1_O0}_vs_{arch2_O3}.BinDiff"
            after_bindiff = f"{arch1_O0_re}_vs_{arch2_O3_re}.BinDiff"

            before_bindiff_path = os.path.join(abs_all_path, before_bindiff)
            after_bindiff_path = os.path.join(abs_all_path, after_bindiff)

            before_bindiff_result = read_one_result_file(before_bindiff_path)
            after_bindiff_result = read_one_result_file(after_bindiff_path)
            inc = (after_bindiff_result - before_bindiff_result) / before_bindiff_result

            arch1_name = arch_mapping[arch1]
            arch2_name = arch_mapping[arch2]

            data[(arch2_name, "Before")][
                index.index(arch1_name)
            ] = before_bindiff_result
            data[(arch2_name, "After")][index.index(arch1_name)] = after_bindiff_result
            data[(arch2_name, "Inc")][index.index(arch1_name)] = inc

    for arch_name in index:
        data[(arch_name, "Before")][4] = np.mean(data[(arch_name, "Before")][0:4])
        data[(arch_name, "After")][4] = np.mean(data[(arch_name, "After")][0:4])
        data[(arch_name, "Inc")][4] = np.mean(data[(arch_name, "Inc")][0:4])

    for i in range(5):
        data[("AVG", "Before")][i] = np.mean(
            [
                data[("ARM", "Before")][i],
                data[("PowerPC", "Before")][i],
                data[("MIPS", "Before")][i],
                data[("X86", "Before")][i],
            ]
        )
        data[("AVG", "After")][i] = np.mean(
            [
                data[("ARM", "After")][i],
                data[("PowerPC", "After")][i],
                data[("MIPS", "After")][i],
                data[("X86", "After")][i],
            ]
        )
        data[("AVG", "Inc")][i] = (
            data[("AVG", "After")][i] - data[("AVG", "Before")][i]
        ) / data[("AVG", "Before")][i]

    return data


def display_result(data):
    # Create the data

    index = ["ARM", "PowerPC", "MIPS", "X86", "AVG"]

    # Create the DataFrame
    df = pd.DataFrame(data, index=index)

    # Style the DataFrame
    styled_df = df.style.set_caption("Table: Performance Comparison")
    styled_df = styled_df.format(precision=4)
    styled_df = styled_df.set_properties(**{"text-align": "center"})
    styled_df = styled_df.set_table_styles(
        [dict(selector="th", props=[("text-align", "center")])]
    )
    styled_df.to_excel("result.xlsx")

    # plot the figure
    before_score_list = []
    after_score_list = []
    name_list = []
    for arch1 in index[:4]:
        for arch2 in index[:4]:
            before_score = data[(arch2, "Before")][index.index(arch1)]
            after_score = data[(arch2, "After")][index.index(arch1)]
            before_score_list.append(before_score)
            after_score_list.append(after_score)
            name_list.append(f"{arch1}-O0 vs {arch2}-O3")

    x_ray = np.arange(16)
    bar_width = 0.37
    plt.figure(figsize=(25, 5))
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["font.family"] = "DejaVu Serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]
    plt.rcParams["font.size"] = 13.5
    b1 = plt.bar(
        x_ray,
        before_score_list,
        bar_width,
        align="center",
        color="#414c87",
        label="Before",
    )
    plt.bar_label(b1, labels=[round(_, 2) for _ in before_score_list], padding=3)
    b2 = plt.bar(
        x_ray + bar_width,
        after_score_list,
        bar_width,
        color="#af5a76",
        align="center",
        label="After",
    )
    plt.bar_label(b2, labels=[round(_, 2) for _ in after_score_list], padding=3)

    plt.legend()
    plt.ylabel("similarity score")
    plt.xticks(x_ray + bar_width / 2, name_list, rotation=45)
    plt.margins(0.1)
    plt.subplots_adjust(bottom=0.4)

    plt.savefig("result.pdf")
    print(
        "Result Excel File is located in result.xlsx, Result Figure is located in result.pdf"
    )


if __name__ == "__main__":
    data_path = args.data_path

    generate_essential_files()
    generate_compare_database(data_path)
    data = read_all_result_files(data_path)
    display_result(data)
