from tqdm import tqdm
import wandb

PROJECT_NAME = "mulm"
KEYS = ['generated_snts'] #,'ordered_ppls']
wandb.login()
api = wandb.Api(overrides={"project": PROJECT_NAME, "entity": "liir-kuleuven"},timeout=1000)
runs = api.runs(f"{PROJECT_NAME}")
dct = {}
pbar = tqdm(runs, total=len(runs))
for i,run in enumerate(pbar):
    if i < 360:
        continue
    pbar.set_description(f"{run.id}")
    # artifacts = run.logged_artifacts()
    # subdct = {}
    # for a in artifacts:
    #     base_name = a.name.split(':')[0]
    #     if base_name not in subdct:
    #         subdct[base_name] = []
    #     subdct[base_name].append(a)
    # for k, lst in subdct.items():
    #     if len(lst) > 1 and any([s in k for s in KEYS]):
    #         v_and_a = [(el.source_version, el.aliases) for el in  lst]
    #         # The highest version (eg v10) should be the only one with aliases, ie the aliases ['latest'].
    #         sorted_v_and_a = sorted(v_and_a, key=lambda x: int(x[0].split('v')[-1]))
    #         assert sorted_v_and_a[-1][1] == ['latest']
    #         assert all([el[1] == [] for el in sorted_v_and_a[:-1]])
    #         # Remove all but the highest version.
    #         for artifact in tqdm(lst[:-1]):
    #             if len(artifact.aliases) == 0:
    #                 artifact.delete()
    # dct[run.id] = subdct
    files = run.files()
    for KEY in KEYS:
        subfiles = [f for f in files if KEY in f._name]
        if len(subfiles) > 1:
            # file names look like f'media/table/{KEY}_{log_step}_{some_hash}.table.json'. We want to keep the last one, ie the one with the highest log_step.
            sorted_files = sorted(subfiles, key=lambda x: int(x._name.split(f'media/table/{KEY}_')[1].split('_')[0]))
            nonlast_subfiles = sorted_files[:-1]
            # assert last file has the biggest size
            if not all([sorted_files[-1].size > f.size for f in nonlast_subfiles]):
                offending_filenames = {f._name:f.size for f in nonlast_subfiles if sorted_files[-1].size <= f.size}
                print(f"Skipping {run.id}, last file + size was {(sorted_files[-1]._name,sorted_files[-1].size)}, offending files and sizes: {offending_filenames}")
                continue
            # assert all([sorted_files[-1].size > f.size for f in nonlast_subfiles]), f"For run {run.id} " + str({f._name:f.size for f in sorted_files})
            for f in tqdm(nonlast_subfiles):
                f.delete()







print(5)
