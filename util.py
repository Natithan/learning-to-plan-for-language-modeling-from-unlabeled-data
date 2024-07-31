import sys
from time import time
from os.path import join as jn
from glob import glob
import subprocess
from constants import DEFAULT_PICKLE_DIR
import pickle
import random
import bisect
def plot_tensor_as_img(tensor,savepath=None, title=None):
    if savepath is None:
        import matplotlib;
        matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    t = fix_tensor(tensor)
    im = plt.imshow(t)
    if title is not None:
        plt.title(title)
    if savepath is None:
        plt.show()
    else:
        plt.savefig(savepath)


def fix_tensor(tensor,vid=False):
    import numpy as np
    # Check if type is numpy
    if type(tensor) != np.ndarray:
        t = tensor.detach().cpu().numpy()
    else:
        t = tensor
    if not vid:
        if t.shape[0] == 3:
            t = t.transpose(1, 2, 0)
    else:
        if t.shape[1] == 3:
            t = t.transpose(0, 2, 3, 1)
    if -1.1 < t.min() < -.9 and .6 < t.max() < 1.1:
        print("Scaling from [-1,1] to [0,1]")
        t = (t + 1) / 2
    if vid:
        t = (t * 255).astype(np.uint8)
    return t


def plot_tensors_as_stacked_imgs(*tensors, horizontal=True, savepath=None, title=None):
    '''
    Plots list of batch tensors as 2D image grid. If horizontal is True, the images are stacked horizontally along the list, and vertically along the batch dimension.
    If vertical is True, vice versa.
    Also works for non-batch tensors.
    '''
    import torch
    if len(tensors[0].shape) == 3:
        tensors = [t[None] for t in tensors]
    stacked_tensor = torch.concat([t.permute(1, 2, 0, 3).flatten(-2, -1) for t in tensors], dim=1 if horizontal else 2) # Didn't test vertical :P
    plot_tensor_as_img(stacked_tensor, savepath=savepath, title=title)


def plot_tensor_as_vid(tensor,pause_time=.2, savepath=None):
    if savepath is None:
        import matplotlib;
        matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    tensor = fix_tensor(tensor,vid=True)
    if savepath is None:
        first_frame = tensor[0]
        im = plt.imshow(first_frame)
        for frame in tensor:
            im.set_data(frame)
            plt.pause(pause_time)
            plt.show()
    else:
        import imageio
        imageio.mimsave(savepath, tensor)


def tn(t):
    if type(t).__name__ == 'Tensor':
        return t.detach().cpu().numpy()
    elif type(t).__name__ == 'ArrayImpl':
        import numpy
        return numpy.array(t)
    elif type(t).__name__ == 'ndarray':
        print("Was already numpy array")
        return t



def count_params(model,trainable_only=False):
    return sum(p.numel() for p in model.parameters() if (p.requires_grad or not trainable_only))


def print_param_count(pl_module, prefix = ""):
    print(prefix,f'All: {count_params(pl_module):,}',f'Trainable: {count_params(pl_module,trainable_only=True):,}')

def summ(tensor):
    import numpy as np
    np_tensor = tn(tensor) if type(tensor) != np.ndarray else tensor
    print("shape: ", np_tensor.shape)
    print("min/max: ", np_tensor.min(), np_tensor.max())
    print("mean: ", np_tensor.mean())
    print("std: ", np_tensor.std())


def print_minmax_multiindex(tensor, n=1):
    import numpy as np
    np_tensor = tn(tensor) if type(tensor) != np.ndarray else tensor
    # print("min/max: ", np_tensor.min(), np_tensor.max())
    # print("argmin/argmax: ", np.unravel_index(np_tensor.argmin(), np_tensor.shape), np.unravel_index(np_tensor.argmax(), np_tensor.shape))
    print("min")
    # smallest n values
    for i in np.argpartition(np_tensor.flatten(), n)[:n]:
        print(np.unravel_index(i, np_tensor.shape), np_tensor.flatten()[i])
    print("max")
    # largest n values
    for i in np.argpartition(np_tensor.flatten(), -n)[-n:]:
        print(np.unravel_index(i, np_tensor.shape), np_tensor.flatten()[i])


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    # | operator: if doing namespace1 |= namespace2, update namespace1 with namespace2. If n1 | n2, return a new namespace with n1 and n2
    def __or__(self, other):
        if type(other) == Namespace:
            return Namespace(**{**self.__dict__, **other.__dict__})
        else:
            raise ValueError("Can only | with another Namespace")

    # allow ** unpacking
    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __delitem__(self, key):
        del self.__dict__[key]

    def __repr__(self):
        return str(self.__dict__)

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __iter__(self):
        return iter(self.__dict__)

    def __len__(self):
        return len(self.__dict__)



def CUDAfy(batch):
    CUDA_DEVICE = 'cuda'
    # if type(batch) == dict:
    #     # dict of lists of tensors to cuda
    #     batch = {k: [el.to(CUDA_DEVICE) for el in v] for k, v in batch.items()}
    # else:
    #     batch = [el.to(CUDA_DEVICE) if el != [] else [] for el in batch]  # can be empty list if require_imgs is False
    if type(list(batch.values())[0]) == dict:
        # dict of dicts of tensors to cuda
        batch = {k: {k2: el2.to(CUDA_DEVICE) for k2, el2 in v.items()} for k, v in batch.items()}
    else:
        batch = {k: el.to(CUDA_DEVICE) if el != [] else [] for k, el in batch.items()}
    return batch


def get_codebook(codebook_path):
    import pickle
    with open(codebook_path, 'rb') as f:
        print("Loading codebook from", codebook_path)
        s = time()
        loaded = pickle.load(f)
        print(f"Loaded codebook in {time() - s} seconds")
        if 'codebook' in loaded:
            return loaded['codebook']
        else:
            return loaded


def get_logger(args):
    resume = args.pop('resume')
    if resume:
        if args.logger_wid is not None:
            logger_id = args.logger_wid
        elif args.ckpts_wid is not None:
            logger_id = args.ckpts_wid
        elif args.jplm_ckpt_wid is not None:
            logger_id = args.jplm_ckpt_wid
        elif args.plm_ckpt_wid is not None:
            logger_id = args.plm_ckpt_wid
        elif args.mz_ckpt_wid is not None:
            logger_id = args.mz_ckpt_wid
        elif args.lm_ckpt_wid is not None:
            logger_id = args.lm_ckpt_wid
        else:
            raise ValueError("No wid given, but resume=True")
        logger_kwargs = {'resume': 'must', 'id': logger_id}
        allow_val_change=True
    else:
        if args.logger_wid is not None:
            logger_kwargs = {'id': args.logger_wid}
        else:
            logger_kwargs = {}
        allow_val_change=False

    from lightning.pytorch.loggers import WandbLogger
    logger = WandbLogger(project="mulm", notes=args.pop('notes'), tags=args.pop('tags'), log_model=False, **logger_kwargs)
    run = logger.experiment
    cfg = run.config
    if not callable(cfg):
        default_name = run.name
        postfix = args.pop('postfix')
        maybe_run_number = ('-' + default_name.split('-')[-1]) if (len(postfix) == 0) else ''
        if args.name is not None:
            run.name = args.pop('name')
        else:
            run.name = f"{args.pop('prefix')}{short_name(args.base_model_name)}{'-noClstr' if args.no_cluster else ''}{postfix}{maybe_run_number}"
        cfg.update(args, allow_val_change=allow_val_change)
        if not resume:
            cfg['cli_args'] = " ".join(sys.argv[1:])
        else:
            cfg.update({'cli_args': "OLD: " + cfg['cli_args'] + " NEW: " + " ".join(sys.argv[1:])}, allow_val_change=True)
        cfg['effective_lm_batch_size'] = args.batch_size * args.accumulate_lm_grad_batches
        cfg['effective_mzpt_batch_size'] = args.mzpt_batch_size * args.accumulate_mz_grad_batches

        cfg['ckpts_wid'] = args.ckpts_wid
        cfg['lm_ckpt_wid'] = args.lm_ckpt_wid
        cfg['lm_ckpt_path'] = args.lm_ckpt_path
        cfg['mz_ckpt_wid'] = args.mz_ckpt_wid
        cfg['mz_ckpt_path'] = args.mz_ckpt_path

        # cfg['commit_id'] = get_git_commit_id()

    return logger


def short_name(name):
    if name.startswith('meta-llama'):
        return 'llama'
    else:
        return name


def get_ckpt_callbacks(args, logger, monitor='val_loss', prefix=''):
    from lightning.pytorch.callbacks import ModelCheckpoint
    exp_id = logger.experiment.id
    common_args = {'dirpath': f'{args.checkpoint_dir}/{exp_id}', 'save_top_k': args.save_top_k}
    best_args = common_args | {'monitor': monitor, 'filename': prefix + 'BEST-{epoch}-{step}'}
    last_args = common_args | {'monitor': None, 'filename': prefix + 'LAST-{epoch}-{step}'}
    return [ModelCheckpoint(**best_args), # best-ckpt
            ModelCheckpoint(**last_args), # last-ckpt
            ]


# def get_cluster_samples(cluster_idx, book_and_codes_path=None, N=10):
#     if book_and_codes_path is None:
#         book_and_codes_path = jn(DEFAULT_PICKLE_DIR, "wikitext-103_a28531_seq/c1024/book_and_coded_sents.pkl")
#     with open(book_and_codes_path, 'rb') as f:
#         book_and_codes = pickle.load(f)
#     sents = [article['sentences'][i] for article in book_and_codes['articles'] for i, c in enumerate(article['codes'])
#              if c == cluster_idx]
#     if N > 0:
#         try:
#             return random.sample(sents, N)
#         except ValueError as e:
#             if 'Sample larger than population' in str(e):
#                 print(f"Only {len(sents)} samples available for cluster {cluster_idx}")
#                 return sents
#     else:
#         return sents


def scale_gradient(tensor, scale):
    """Scales the gradient for the backward pass."""
    return tensor * scale + tensor.detach() * (1 - scale)

def get_git_commit_id():
    try:
        # Run the Git command to get the latest commit ID
        commit_id = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip()
        return commit_id.decode()
    except subprocess.CalledProcessError as e:
        # Handle errors if Git command fails
        return "Error: " + str(e)


def is_sublist(sublist, list):
    if len(sublist) > len(list):
        return False
    for i in range(len(list) - len(sublist) + 1):
        if list[i:i + len(sublist)] == sublist:
            return True
    return False

def pload(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def pdump(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def is_in_sorted_list(sorted_list, element):
    index = bisect.bisect_left(sorted_list, element)
    return index < len(sorted_list) and sorted_list[index] == element


def get_relevant_ckpt_path(args, trainer):
    last_ckpt_callback = [c for c in trainer.checkpoint_callbacks if 'LAST' in c.filename][0]
    best_ckpt_callback = [c for c in trainer.checkpoint_callbacks if 'BEST' in c.filename][0]
    callback = last_ckpt_callback if args.last_iso_best_ckpt else best_ckpt_callback
    ckpt_path = callback.best_model_path
    return ckpt_path
