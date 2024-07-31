import wandb
import os
import shutil
import re
import argparse
from os.path import join as jn

from constants import CODE_DIR

wandb_username = 'liir_kuleuven'
project_name = 'mulm'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--not_dry_run", action="store_true")
    args = parser.parse_args()
    # API call to get all run ids in a project
    api = wandb.Api()
    runs = api.runs(f"{project_name}")
    online_run_ids = [run.id for run in runs]

    # Path to local wandb directory
    local_wandb_dir = jn(CODE_DIR,"wandb")
    wandb_pattern = r'^run-\d{8}_\d{6}-([a-zA-Z0-9]{8})$'
    # Path to local checkpoints directory
    local_checkpoints_dir = jn(CODE_DIR,"checkpoints")
    checkpoints_pattern = r'^([a-zA-Z0-9]{8})$'


    # Determine which local runs to delete
    to_delete = []
    for local_dir, pattern in zip([local_wandb_dir, local_checkpoints_dir], [wandb_pattern, checkpoints_pattern]):
        dirs = os.listdir(local_dir)
        for subdir in dirs:
            id_or_none = re.match(pattern, subdir)
            if id_or_none is not None:
                local_id = id_or_none.groups()[0]
                if local_id not in online_run_ids:
                    local_run_path = os.path.join(local_dir, subdir)
                    to_delete.append(local_run_path)

    # Print which runs will be deleted
    print(f"Deleting {len(to_delete)} dirs: {to_delete}")
    # calculate how much mb will be deleted
    total_size = 0
    for path in to_delete:
        total_size += sum(os.path.getsize(os.path.join(dirpath, filename)) for dirpath, dirnames, filenames in os.walk(path) for filename in filenames)
    print(f"Total size to be deleted: {total_size / 1e6} MB")

    # Delete local runs
    if args.not_dry_run:
        for path in to_delete:
            print(f"Deleting {path}")
            shutil.rmtree(path)
    else:
        print("Dry run, not deleting anything. Use --not_dry_run to actually delete.")

if __name__ == '__main__':
    main()
