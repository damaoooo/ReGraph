set -e

# source_dir=/home/damaoooo/Downloads/OpenPLC_v3/webserver/core
dest_dir="$(pwd)/openplc_dataset"

# Function to display help message
usage() {
  echo "Usage: $0 -s <dir> [-d <dir>]"
  echo "   or: $0 --source_dir=<dir> [--dest_dir=<dir>]"
  echo "   or: $0 --source_dir <dir> [--dest_dir <dir>]"
  echo
  echo "Options:"
  echo "  -s, --source_dir  The source code directory (required)"
  echo "  -d, --dest_dir    The output directory (default: ./openplc_dataset)"
  exit 1
}

# 解析命令行参数
while [[ "$1" != "" ]]; do
  case $1 in
    -s | --source_dir)
      if [[ "$2" != "" && "$2" != -* ]]; then
        source_dir=$2
        shift
      else
        source_dir="${1#*=}"
      fi
      ;;
    --source_dir=*)
      source_dir="${1#*=}"
      ;;
    -d | --dest_dir)
      if [[ "$2" != "" && "$2" != -* ]]; then
        dest_dir=$2
        shift
      else
        dest_dir="${1#*=}"
      fi
      ;;
    --dest_dir=*)
      dest_dir="${1#*=}"
      ;;
    -h | --help)
      usage
      ;;
    *)
      echo "Unknown parameter: $1"
      usage
      ;;
  esac
  shift
done

# 检查是否提供了 source_dir
if [[ -z "$source_dir" ]]; then
  echo "Error: --source_dir is required."
  usage
fi


# Test whether dest_dir exists, if not, create it
if [ ! -d ${dest_dir} ]
then
    mkdir -p ${dest_dir}
fi
cd ${dest_dir}
rm -rf ./*

cd ${source_dir}
rm -rf Res0_*

for compiler in g++ arm-linux-gnueabi-g++ powerpc-linux-gnu-g++ mips-linux-gnu-g++
do
    for optimize in -O0 -O1 -O2 -O3
    do
        filename=${dest_dir}/Res0_${compiler}${optimize}.o

        if [ ${compiler} == 'g++' ]
        then
            $compiler -std=gnu++11 -m32 -I ${source_dir}/lib -c Res0.c ${optimize} -fno-inline-functions -fno-inline -lasiodnp3 -lasiopal -lopendnp3 -lopenpal -w -o ${filename}
        else
            $compiler -std=gnu++11 -I ${source_dir}/lib -c Res0.c ${optimize} -fno-inline-functions -fno-inline -lasiodnp3 -lasiopal -lopendnp3 -lopenpal -w -o ${filename}
        fi
        retdec-decompiler ${filename} -s -k --backend-keep-library-funcs
        llvm-dis ${filename}.bc -o ${filename}.ll
        clang -m32 -O3 -c ${filename}.ll -fno-inline-functions -o ${filename}_re

        rm ${dest_dir}/*.dsm
        rm ${dest_dir}/*.config.json
        rm ${dest_dir}/*.bc
        rm ${dest_dir}/*.c
        rm ${dest_dir}/*.ll

    done
done
