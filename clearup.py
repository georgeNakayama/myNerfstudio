import os 
import glob
import shutil
from multiprocessing.pool import ThreadPool
path = "/orion/u/w4756677/slurm_dump"
pool = ThreadPool(processes=16)
dirs = glob.glob(path + '/*')
dirs = [d for d in dirs if "llff" in d or "cimle" in d or "test" in d or "cache" in d or "fern" in d or "exp" in d]
print(dirs)
def rm(d):
    try:
        os.remove(d)
        print("File '%s' has been removed successfully" %d)
    except OSError as error:
        print(error)
        print("File '%s' can not be removed" %d)
        
pool.map(rm, dirs)
    