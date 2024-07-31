import matplotlib.pyplot as plt
import matplotlib
import json
import pandas
import os
import wandb


normal_wid = "9zv395lx" #"c5dbag7o"
addrand_wid = "8e69rkse" #"6htj4bjg"

api = wandb.Api()
def get_v(wid):
    return int(api.artifact(f"liir-kuleuven/mulm/run-{wid}-evallm_ordered_ppls:latest").version.split("v")[1])

used_version = min(get_v(normal_wid), get_v(addrand_wid))
print("USING VERSION", used_version)

def get_ppls_per_rank(wid):
    global artifact, file, ranks, ppls
    artifact = api.artifact(f"liir-kuleuven/mulm/run-{wid}-evallm_ordered_ppls:v{used_version}", type='run_table')
    dir = artifact.download()
    path = os.path.join(dir, "eval", "lm_ordered_ppls.table.json")
    with open(path, 'r') as file:
        print("Reading", path)
        obj = json.load(file)
    ranks, ppls = zip(*obj["data"])
    assert ranks == tuple(list(range(len(ranks))))
    return ppls


normal_ppls = get_ppls_per_rank(normal_wid)
addrand_ppls = get_ppls_per_rank(addrand_wid)

def get_value_at_step(wid, key_name, step):
    """Retrieves the value logged under 'key_name' at the specific 'step' in a run."""
    run = api.run(f"liir-kuleuven/mulm/{wid}")  # Get the run object
    history = run.history(keys=[key_name])  # Fetch history of just the key
    return history[key_name][step]  # Access value at the given step

# Example Usage: Get values for each run at step 100
ORACLE_PPL = get_value_at_step(normal_wid, "eval/lm_oracle_perplexity", used_version)
ART_COUNT = get_value_at_step(normal_wid, "eval/lm_article_count", used_version)
print(f"Oracle PPL: {ORACLE_PPL}, Article Count: {ART_COUNT + 1}")

# rank with ppl closest to ORACLE_PPL
# ORACLE_RANK = ranks[abs(ppls - ORACLE_PPL).argmin()]
NORMAL_ORACLE_RANK = normal_ppls.index(min(normal_ppls, key=lambda x: abs(x - ORACLE_PPL)))
ADDRAND_ORACLE_RANK = addrand_ppls.index(min(addrand_ppls, key=lambda x: abs(x - ORACLE_PPL)))
# Plotting
matplotlib.rcParams.update({'font.size': 14})
plt.figure(figsize=(10, 6))
# plt.plot(ranks, ppls, marker='o', linestyle='-', color='b')
line1, = plt.plot(range(len(normal_ppls)), normal_ppls, marker='o', linestyle='-', color='b')
plt.yscale('log')
plt.xlabel('Rank')
plt.ylabel('Perplexity')
plt.title('Average Perplexity at each Rank')
plt.grid(True, which="both", ls="--")
# Red horizontal line at ORACLE_PPL and vertical line at ORACLE_RANK
plt.axhline(y=ORACLE_PPL, color='r', linestyle='--')
plt.axvline(x=NORMAL_ORACLE_RANK, color='b', linestyle='--')
plt.tight_layout()
# plt.text(0 - 10, ORACLE_PPL + 1, f'Oracle: {ORACLE_PPL:.2f}', color='r')
plt.annotate(f'Oracle: {ORACLE_PPL:.2f} ', xy=(-45 , ORACLE_PPL+1), horizontalalignment='right', color='r', fontsize=12)
plt.subplots_adjust(left=0.21)
plt.text(NORMAL_ORACLE_RANK + 10, 40, f'Oracle rank among actions: {NORMAL_ORACLE_RANK}', color='b')
# blue horizontal line at best normal rank + text
plt.axhline(y=normal_ppls[0], color='b', linestyle='--')
# plt.text(0 - 100, normal_ppls[0] + 1, f'Perplexity of best action: {normal_ppls[0]:.2f}', color='b')
plt.annotate(f'Best action: {normal_ppls[0]:.2f} ', xy=(-45 , normal_ppls[0] - 0.5), horizontalalignment='right', color='b', fontsize=12)
plt.savefig('/mnt/g/My Drive/1 PhD/0 muLM/Drawings/better_than_oracle_codes.pdf')



# plt.text(ADDRAND_ORACLE_RANK + 10, 0 + 10, f'{ADDRAND_ORACLE_RANK}', color='g')
plt.axvline(x=ADDRAND_ORACLE_RANK, color='g', linestyle='--')
plt.text(ADDRAND_ORACLE_RANK + 10, 50, f'Oracle rank among noise variations: {ADDRAND_ORACLE_RANK}', color='g')
line3, = plt.plot(range(len(addrand_ppls)), addrand_ppls, marker='o', linestyle='-', color='g')
# green horizontal line at best addrand rank + text
plt.axhline(y=addrand_ppls[0], color='g', linestyle='--')
# plt.text(0 - 100, addrand_ppls[0] + 1, f'Best noisy variant: {addrand_ppls[0]:.2f}', color='g')
plt.annotate(f'Best noise variant: {addrand_ppls[0]:.2f} ', xy=(-45 , addrand_ppls[0]), horizontalalignment='right', color='g', fontsize=12)
plt.legend(handles=[line1, line3],labels=['Learnt action embeddings', 'Noisy variants of oracle embedding'])
plt.savefig('/mnt/g/My Drive/1 PhD/0 muLM/Drawings/better_than_oracle_codes_2.pdf')
print("Done")



