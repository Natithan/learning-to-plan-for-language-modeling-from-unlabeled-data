# region preamble
from time import time
from pretorch_util import assign_visible_gpus
assign_visible_gpus()
print("importing stuff"); s = time()
import warnings
warnings.filterwarnings("ignore", message="Using `TRANSFORMERS_CACHE` is deprecated")
warnings.filterwarnings("ignore", message="TypedStorage is deprecated.")
from tune_planner import get_oracle_pretrained_planner
from eval import eval_planner_lm_combo, eval_ppl_parallel, noctx_geneval, startfullctx_geneval
from args import get_args
from create_oracle_codes import get_ds_with_oracle_codes
from tune_cclm import get_tuned_cclm
from util import get_logger

print(f"imported stuff in {time()-s:.2f}s")
warnings.filterwarnings("ignore", ".*does not have many workers.*") # At time of writing (22 dec '23), increasing num_workers does not increase speed, so this warning is not relevant
# endregion


def main():
    """
    script that can combine
    - initial code creation + pickling
    - oracle-code-condtioned LM training
    - oracle-muzero training
    (later: non-oracle LM and muzero training)
    """
    args = get_args()
    ds = get_ds_with_oracle_codes(args)
    logger = get_logger(args)

    language_model = get_tuned_cclm(args, dataset=ds, logger=logger) if not args.skip_lm_load else None
    planner = get_oracle_pretrained_planner(args, dataset=ds, language_model=language_model, logger=logger) if not args.skip_mz_load else None
    if args.tune_lm_with_planner:
        language_model = get_tuned_cclm(args, dataset=ds, logger=logger, planner=planner, initial_model=language_model, joint_planner_lm=False) if not args.skip_plm_load else None
    if args.joint_planner_lm:
        language_model = get_tuned_cclm(args, dataset=ds, logger=logger, planner=planner, initial_model=language_model, joint_planner_lm=True)
        if not args.uniform_mix:
            planner = language_model.planner # might not actually be necessary as planner gets updated in-place apparently

    if args.tune_lm_with_planner or args.joint_planner_lm:
        trainer = language_model._trainer
    elif not args.skip_mz_load:
        trainer = planner._trainer
    elif not args.skip_lm_load:
        trainer = language_model._trainer
    else:
        trainer = None
    if trainer is not None:
        trainer.strategy.barrier()
        if trainer.global_rank > 0:
            exit(0)

    if args.parallel_ppl_eval:
        eval_ppl_parallel(args, ds, logger, language_model, planner)

    if args.startfullctx_geneval:
        startfullctx_geneval(args, ds, logger, language_model, planner)

    if args.noctx_geneval:
        noctx_geneval(args, ds, logger, language_model, planner)

    if not args.skip_eval:
        if not args.nll_then_lev:
            eval_planner_lm_combo(args, ds, logger, language_model, planner)
        else:
            print("Setting skip_nll_eval to False and structure_levenshtein_eval to False")
            args.skip_nll_eval = False
            args.structure_levenshtein_eval = False
            eval_planner_lm_combo(args, ds, logger, language_model, planner, stage='_nll')
            print("Setting skip_nll_eval to True and structure_levenshtein_eval to True")
            args.skip_nll_eval = True
            args.structure_levenshtein_eval = True
            eval_planner_lm_combo(args, ds, logger, language_model, planner, stage='_lev')


if __name__ == '__main__':
    main()


