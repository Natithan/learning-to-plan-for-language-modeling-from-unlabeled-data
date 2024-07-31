import os
import re
import sys

def get_free_gpus(free_mem_based=False):
    # os.popen('gpustat --no-header').readlines() returns a list of strings, each string is a line like this:
    # "[0] NVIDIA GeForce GTX TITAN X | 50'C,   0 % |  1298 / 12288 MB | nathan(1214M) "
    # regex to extract used memory, total memory and process names
    # so from "[0] NVIDIA GeForce GTX TITAN X | 50'C,   0 % |  1298 / 12288 MB | nathan(1214M) ", extract 1298, 12288 and nathan(1214M)
    regex = r"% \|.*?(\d+) / (\d+) MB \|(.*?)\n"
    processes = [re.search(regex, l) for l in os.popen('gpustat --no-header').readlines()]
    free_mem = [int(p.groups()[1]) - int(p.groups()[0]) if p is not None else 0 for p in processes]
    if not free_mem_based: # gpu based on name
        free_gpus = [i for i, p in enumerate(processes) if p is not None and
                     all([('nathan' in s) for s in p.groups()[2].split()])]  # gpus where only my processes are running
    else:
        # gpu not based on name, but just on most free MB of memory
        free_gpus = [i for i, m in enumerate(free_mem) if m == max(free_mem)]
    return free_gpus, free_mem

def get_user_gpu_mem_usage():
    # os.popen('gpustat --no-header').readlines() returns a list of strings, each string is a line like this:
    # "[0] NVIDIA GeForce GTX TITAN X | 50'C,   0 % |  1298 / 12288 MB | nathan:python/2483(1214M) "
    # regex to extract memory usage for user, in the above example, 1214. The process id above is 2483
    pid = os.getpid()
    regex = fr".* MB \|.*{pid}\((\d+)M\).*\n"
    processes = [re.search(regex, l) for l in os.popen('gpustat --no-header -p').readlines()] # p to show process id
    return [int(p.groups()[0]) if p is not None else 0 for p in processes]

def get_really_free_gpus():  # Not even my processes
    regex = r"MB \|(.*?)\n"
    processes = [re.search(regex, l) for l in os.popen('gpustat --no-header').readlines()]
    # free_gpus = [i for i, p in enumerate(processes) if p is not None and
    #              all([('nathan' in s) for s in p.groups()[0].split()])]
    free_gpus = [i for i, p in enumerate(processes) if p is not None and
                 (len(p.groups()[0].split()) == 0)]
    return free_gpus


def rank_to_device(rank):
    '''
    For when not all GPUs on the device can be used
    '''
    free_gpus, _ = get_free_gpus()
    assert len(free_gpus) > rank, "Rank larger than number of available GPUs"
    return free_gpus[rank]


def assign_visible_gpus(free_mem_based=True):
    arg = '--visible_gpus' if '--visible_gpus' in sys.argv else '--gpus'
    if arg in sys.argv and (gpu_arg := sys.argv[sys.argv.index(arg) + 1]) and not gpu_arg.startswith('-'):
        # if len(gpu_arg) > 1:
        #     raise NotImplementedError("Multi-GPU training not supported until figured out why it damages performance for same effective batch size")
        print("=" * 10, f"Manually setting os.environ['CUDA_VISIBLE_DEVICES'] to {gpu_arg}",
              "=" * 10)
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_arg
        del sys.argv[sys.argv.index(arg):sys.argv.index(arg) + 2]
    else:
        if 'CUDA_VISIBLE_DEVICES' in os.environ:
            # print(os.environ['CUDA_VISIBLE_DEVICES'])
            pass
        else:
            print("os.environ['CUDA_VISIBLE_DEVICES'] not set")
            free_gpus, free_mem = get_free_gpus(free_mem_based=free_mem_based)
            # if gpu_arg is smt like -2, then use max 2 gpus
            if arg in sys.argv and (gpu_arg := sys.argv[sys.argv.index(arg) + 1]) and gpu_arg.startswith('-'):
                num_gpus = int(gpu_arg[1:])
                if num_gpus > 1:
                    raise NotImplementedError("Multi-GPU training not supported until figured out why it damages performance for same effective batch size")
                free_gpus = free_gpus[:num_gpus]
                del sys.argv[sys.argv.index(arg):sys.argv.index(arg) + 2]
            free_gpus = free_gpus[:1]  # Only use one GPU
            os.environ['CUDA_VISIBLE_DEVICES'] = ",".join([str(i) for i in free_gpus])
            print(f"Now set to {os.environ['CUDA_VISIBLE_DEVICES']} with free memory {[free_mem[i] for i in free_gpus]}")


