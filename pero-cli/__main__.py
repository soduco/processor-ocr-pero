from argparse import ArgumentParser
import pathlib
import glob
import subprocess 
from tqdm import tqdm
from .process import PeroOCREngine





parser = ArgumentParser(description='PERO OCR command line argument')
parser.add_argument("-i", "--input-dir", required=True, type=pathlib.Path)
parser.add_argument("-o", "--output-dir", required=True, type=pathlib.Path)
parser.add_argument("-f", "--export-format", required=True, choices=["json", "alto", "page", "image"], action="append")
parser.add_argument("--pero-config-file", default="/tmp/pero-printed_modern-public-2022-11-18/config_cpu.ini", type=pathlib.Path)


args = parser.parse_args()
input_images = sorted(glob.glob("*.jpg", root_dir=args.input_dir))
input_jsons = sorted(glob.glob("*.json", root_dir=args.input_dir))

# Sanity check
stems = list(map(lambda x: pathlib.Path(x).stem, input_images))
stems_ = list(map(lambda x: pathlib.Path(x).stem, input_jsons))
if stems != stems_:
    print(stems)
    print(stems_)
    raise RuntimeError("JSON and image lists mismatch.")

# Check config file
cf : pathlib.Path = args.pero_config_file 
if not cf.exists():
    cf.parent.mkdir(parents=True, exist_ok=True)
    tmpdir = cf.parent
    zipfile = cf.with_name("pero-printed_modern-public-2022-11-18.zip")
    cmd = ["wget", "https://www.lrde.epita.fr/~jchazalo/SHARE/pero-printed_modern-public-2022-11-18.zip", "-P", str(tmpdir)] 
    subprocess.call(cmd)
    cmd = ["unzip", str(zipfile), "-d", str(tmpdir)]
    subprocess.call(cmd)

# Create output dir
args.output_dir.mkdir(exist_ok=True)    

## Run
pero = PeroOCREngine(str(cf))
for ima, js in tqdm(zip(input_images, input_jsons)):
    pero.process(args.input_dir / ima, args.input_dir / js, args.output_dir, exports=args.export_format)














