from typing import Any, Dict
from time import time
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from constants import SHORT2FULL_EMBEDDER_NAME
import lightning.pytorch as pl
import torch.nn.functional as F

from transformers import AutoModelForCausalLM

from muzero_tree import MuZeroTree
from misc import muzero_beamsearch, muzero_beamsearch_batched, muzero_beamsearch_batched_vectorized
from util import scale_gradient
from sentence_transformers import SentenceTransformer
import numpy as np
from util import tn
from functools import partial
from tqdm import tqdm as std_tqdm
tqdm = partial(std_tqdm, dynamic_ncols=True)

def scalar_loss(pred, target, reduction='mean'):
    # simple MSE.
    mse = ((pred - target) ** 2)
    if reduction == 'mean':
        return mse.mean()
    elif reduction == 'sum':
        return mse.sum()
    elif reduction == 'none':
        return mse
    else:
        raise ValueError(f'Unknown reduction {reduction}.')

def bce_loss(pred, target, reduction='mean'):
    """
    Note: target input is - NLL, so average loglikelihood. Calling torch.exp() on it will give us the probability of the target.
    """
    # Convert NLL to probability
    target_prob = torch.exp(target)
    assert torch.all(target_prob >= 0) and torch.all(target_prob <= 1), "Target probabilities must be between 0 and 1."
    # Apply sigmoid to predictions
    pred_prob = torch.sigmoid(pred) # does it make sense to use sigmoid here?
    # Compute binary cross entropy
    return nn.BCELoss(reduction=reduction)(pred_prob, target_prob)

class PolicyRegressionLoss(nn.Module):
    def __init__(self, codebook, codebook_transform = None, reduction='mean', ignore_index=-1, distance_function='cosine', negative_sampling='random'):
        super().__init__()
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.codebook = torch.tensor(codebook, device = 'cpu') # make sure it's on CPU cause it could be large!
        self.distance_function = distance_function
        self.codebook_transform = codebook_transform
        self.negative_sampling = negative_sampling
        self.negative_sampling_temperature = 1.0

    def _compute_contrastive_loss(self, input1, input2, negatives):
        positive_loss = torch.nn.CosineEmbeddingLoss(reduction='none')(input1, input2, torch.ones(input1.size(0)).long().to(input1.device))
        # Process negative examples
        negative_loss = torch.nn.CosineEmbeddingLoss(reduction='none')(input1, negatives, -torch.ones(input1.size(0)).long().to(input1.device))
        return positive_loss + negative_loss

    def forward(self, pred, target):
        # pred is an embedding, target is a one hot vector
        # first need to transform one-hot-vector into embedding to compare with pred
        target_embedding = self.codebook[target.cpu()]
        target_embedding = target_embedding.to(pred.device)

        if self.codebook_transform is not None:
            self.target_embedding = self.codebook_transform(target_embedding)

        # compute distance between pred and target
        if self.distance_function == 'euclidean':
            distance = torch.cdist(pred, target_embedding)
        elif self.distance_function == 'cosine':
            distance = 1 - torch.nn.functional.cosine_similarity(pred, target_embedding, dim=-1)
        elif self.distance_function == 'contrastive':
            # Prepare negatives
            if self.negative_sampling == 'random':
                randint = torch.randint(0, self.codebook.size(0), (pred.size(0),))
                negatives = self.codebook[randint]
            elif self.negative_sampling == 'weighted_hard_negatives':
                # Compute softmax over the negative distances
                softmax_distances = torch.nn.functional.softmax(distance / self.negative_sampling_temperature, dim=-1)
                # Sample an index from that distribution
                sampled_indices = torch.multinomial(softmax_distances.squeeze(), 1).squeeze()
                negatives = self.codebook[sampled_indices]
            else:
                raise ValueError(f'Unknown negative sampling method {self.negative_sampling}.')
            # Compute contrastive loss
            negatives = negatives.to(pred.device)
            if self.codebook_transform is not None:
                negatives = self.codebook_transform(negatives)
            distance = self._compute_contrastive_loss(pred, target_embedding, negatives)
        else:
            raise ValueError(f'Unknown distance function {self.distance_function}.')

        if self.reduction in ['mean', 'sum']:

            # in this case we need to compute a mask by checking which of the targets have ignore_index
            mask = target != self.ignore_index
            distance = distance * mask.unsqueeze(-1)
            distance = distance.sum()

            # now divide by the number of non-ignored targets
            if self.reduction == 'mean':
                distance = distance / (mask.sum() * pred.size(-1))
            return distance

        elif self.reduction == 'none':
            return distance
        else:
            raise ValueError(f'Unknown reduction {self.reduction}.')

def is_embedding_space_prediction(policy_loss_fn):
    return policy_loss_fn in ['mse', 'cosine', 'contrastive']

def compute_distances_to_codebook(pred_policy, codebook, distance_function='cosine'):
    # pred_policy: [batch_size, emb_size]
    # codebook: [num_codes, emb_size]
    p = pred_policy.to(codebook.device)
    if distance_function == 'euclidean':
        distances = torch.cdist(p, codebook)
    elif distance_function in ['cosine', 'contrastive']:
        distances = 1 - torch.nn.functional.cosine_similarity(p.unsqueeze(1), codebook.unsqueeze(0), dim=-1)
    else:
        raise ValueError(f'Unknown distance function {distance_function}.')
    return distances.to(pred_policy.device)


class CodebookTransformNet(nn.Module):
    def __init__(self, emb_size, use_mlp=True, use_residual=True):
        super().__init__()
        self.use_mlp = use_mlp
        self.use_residual = use_residual
        if self.use_mlp:
            self.net = nn.Sequential(
                nn.Linear(emb_size, emb_size, bias=False),
                nn.ReLU(),
                nn.Linear(emb_size, emb_size, bias=False)
            )
        else:
            self.net = nn.Linear(emb_size, emb_size, bias=False)

    def forward(self, x):
        residual = x

        x = self.net(x)
        if self.use_residual:
            x = x + residual
        return x

def get_regression_policy_logits(pred_policy, codebook, codebook_transform, distance_function, device=None):

    in_device = pred_policy.device
    if device is not None:
        pred_policy = pred_policy.to(device)
        codebook = codebook.to(device)

    if codebook_transform is not None:
        with torch.no_grad():
            cb = codebook_transform(codebook.to(pred_policy.device))
    else:
        cb = codebook
    pred_policy = compute_distances_to_codebook(pred_policy, cb, distance_function=distance_function) # [batch_size, num_codes]
    # the smaller the distance, the better
    pred_policy = -pred_policy
    return pred_policy.to(in_device)


class MuZero(MuZeroTree, pl.LightningModule): # (note that order of parents is important here)
    def __init__(self, model_h, model_g, model_f, language_model, num_timesteps, mz_multi_vector_representation, policy_only, *args, **kwargs):
        super().__init__(model_h, model_g, model_f, num_timesteps, multi_vector_representation = mz_multi_vector_representation, only_policy_head=policy_only)
        self.save_hyperparameters(ignore=['model_h', 'model_g', 'model_f', 'language_model'])
        if not self.hparams.only_policy_head:
            self.lm = language_model # TODO If resuming halfway mzpt with NOT only_policy_head, there will be a missingkey exception because I don't store the LM as a checkpoint. To solve it, should either call checkpoint loading with strict = False (but then it seems more error prone), or find another way to ensure it doesn't try to load the LM from checkpoint. I'm leaving this problem to future me because currently we're working withing only_policy_head=True anyway :P

        if self.hparams.only_policy_head:
            if self.hparams.mz_policy_loss_fn == 'ce':
                self.policy_loss_fn = CrossEntropyLoss(ignore_index=-1, reduction='none')
            elif self.hparams.mz_policy_loss_fn == 'mse':
                if self.hparams.mz_regression_loss_transform == "mlp":
                    codebook_transform = CodebookTransformNet(self.lm.hparams.codebook.shape[1], use_mlp=True)
                elif self.hparams.mz_regression_loss_transform == "linear":
                    codebook_transform = CodebookTransformNet(self.lm.hparams.codebook.shape[1], use_mlp=False)
                elif self.hparams.mz_regression_loss_transform is None:
                    codebook_transform = None
                else:
                    raise ValueError(f'Unknown codebook transform {self.hparams.mz_regression_loss_transform}.')
                self.policy_loss_fn = PolicyRegressionLoss(self.lm.hparams.codebook, codebook_transform=codebook_transform,
                                                            reduction='mean', ignore_index=-1,
                                                            distance_function = self.hparams.mz_policy_loss_fn_distance_function)
        else:
            if self.hparams.mz_policy_loss_fn == 'ce':
                self.policy_loss_fn = nn.CrossEntropyLoss(ignore_index=-1, reduction='none')
            elif self.hparams.mz_policy_loss_fn == 'mse':
                #self.policy_loss_fn = PolicyRegressionLoss(self.lm.hparams.codebook, codebook_transform=None, reduction='none', ignore_index=-1)
                raise NotImplementedError()


    def training_step(self, batch, batch_idx):
        return self._split_step(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx):
        return self._split_step(batch, batch_idx, 'val')

    def test_step(self, batch, batch_idx):
        return self._split_step(batch, batch_idx, 'test')

    def _split_step(self, batch, batch_idx, split):
        # unpack batch
        attention_mask, snt_idxs, label_mask, context_mask = [batch[k] for k in ['attention_mask', 'snt_idxs', 'label_mask', 'ctx_attention_mask']]

        # num_steps to predict is determined by the batch element with most sentences (count of unique snt_idxs)
        snt_idxs_compressed = [list(sorted(set([idx for idx in snt_idxs[i].tolist() if idx != -1]))) for i in range(len(snt_idxs))] # Excluding -1 for cc_type == insert
        max_steps_data = max([len(snt_idxs_compressed[i]) for i in range(len(snt_idxs_compressed))])

        if 'input_ids_and_codes' in batch:
            codes_between_tokens = True
            input_ids_and_codes = batch['input_ids_and_codes']
            code_mask = batch['code_mask']
            oracle_codes = [input_ids_and_codes[i:i+1][code_mask[i:i+1]] for i in range(len(input_ids_and_codes))]
            target_codes = self.get_target_codes(codes_between_tokens, oracle_codes, snt_idxs)
            context_ids_and_codes, context_code_mask = [batch[k] for k in ['ctx_input_ids_and_codes', 'ctx_code_mask']]
            planner_args = {
                'input_ids': context_ids_and_codes,
                'attention_mask': torch.logical_and(batch['ctx_attention_mask'], ~context_code_mask), # don't attend to code tokens
            }
            actionget_args = {
                'input_ids_and_codes': input_ids_and_codes,
                'code_mask': code_mask,
            }

        else:
            codes_between_tokens = False
            input_ids = batch['input_ids']
            oracle_codes = batch['codes']
            target_codes = self.get_target_codes(codes_between_tokens, oracle_codes, snt_idxs)
            context_ids, context_codes = \
                [batch[k] for k in ['ctx_input_ids', 'ctx_codes']]
            planner_args = {
                'input_ids': context_ids,
                'attention_mask': context_mask,
            }
            actionget_args = {
                'target_codes': target_codes,
                'oracle_codes': oracle_codes,
                'snt_idxs': snt_idxs,
                'snt_idxs_compressed': snt_idxs_compressed,
            }
            if self.hparams.soft_planner_targets:
                actionget_args['split'] = batch['split'][0] # makes difference between eg esval and val

        

        num_steps = min(self.hparams.mz_max_timesteps, max_steps_data)
        mask, oracle_actions, targets = self.get_actions_and_mask(**actionget_args, num_steps=num_steps)
        # ids_for_model_h = torch.cat([context_ids, input_ids], dim=1) if self.hparams.only_policy_head else context_ids
        predictions = self(**planner_args, actions=targets, num_timesteps=num_steps, teacher_forcing=split == 'train') # shape (batch_size, num_timesteps, ...).

        if self.hparams.only_policy_head:
            pred_policy = predictions # [batch_size, seq_len, num_codes]
            #if codes_between_tokens:
            #    assert (code_mask[:,0] == True).all()
            #    next_actions = input_ids_and_codes[:,0]
            #else:
            #    next_actions = target_codes[:,:num_steps]
            #policy_loss = self.policy_loss_fn(pred_policy, next_actions.to(torch.long))


            if self.hparams.soft_planner_targets:
                policy_loss = nn.KLDivLoss(reduction="none")(F.log_softmax(pred_policy, dim=-1), targets.view(*pred_policy.shape))
            else:
                policy_loss = self.policy_loss_fn(pred_policy.view(-1, pred_policy.shape[-1]),targets.view(-1).to(torch.long)).view(pred_policy.shape[:-1])
            # for some reason policy_loss is missing a dimension. Nathan: seems like predictions is missing the num_timesteps dimension if num_steps == 1
            if policy_loss.dim() == 1:
                policy_loss = policy_loss.unsqueeze(-1)
                pred_policy = pred_policy.unsqueeze(1)
            loss = policy_loss
            numerator = (loss * mask.to(loss.device)).sum() 
            denominator = mask.sum().float()
            loss = numerator / denominator
            
            reward_loss, value_loss, next_actions = None, None, None
        else:
            if codes_between_tokens or self.hparams.soft_planner_targets:
                raise NotImplementedError("TODO implement cc_type=insert/soft_planner_targets for non-only_policy_head")
            pred_policy, pred_value, pred_reward = predictions
            mask, oracle_actions, targets = self.get_actions_and_mask(**actionget_args, num_steps=num_steps)
            policy_loss = self.policy_loss_fn(pred_policy.view(-1, pred_policy.shape[-1]), targets.view(-1).to(torch.long)).view(pred_policy.shape[:-1])

            # We DO want to take context_ids into account actually
            def apply_lm_with_codes(lm_codes):
                context_and_target_input_ids = torch.cat([context_ids, input_ids], dim=1)
                context_and_target_attention_mask = torch.cat([context_mask, attention_mask], dim=1)
                with torch.no_grad():
                    self.lm.eval()
                    lm_output = self.lm(context_and_target_input_ids, codes=lm_codes, attention_mask=context_and_target_attention_mask)
                num_target_ids = input_ids.shape[1]
                shift_logits = lm_output.logits[..., -num_target_ids:-1, :].contiguous()
                masked_labels = torch.masked_fill(input_ids, ~label_mask, -100)
                shift_labels = masked_labels[..., 1:].contiguous()
                nll_per_token = CrossEntropyLoss(reduction='none')(shift_logits.view(-1, shift_logits.shape[-1]), shift_labels.view(-1)).view(shift_labels.shape)

                # We now need to convert the per-token nll into a per-sentence nll to form the basis for the target reward and target value.
                nll_per_snt = []
                shift_snt_idxs = snt_idxs[..., :-1].contiguous()
                for batch_idx in range(input_ids.shape[0]):
                    nll_per_snt.append([])
                    for snt_idx in sorted(set(shift_snt_idxs[batch_idx].tolist())):
                        relevant_idxs = torch.logical_and(shift_snt_idxs[batch_idx] == snt_idx, shift_labels[batch_idx] != -100)
                        if relevant_idxs.sum() == 0:
                            # This can happen if there is a sentence with one token. Because we only use the target tokens for calculating the nll targets (so not conditioned on context tokens: we do this for speed), this sentence will have no target tokens and thus no nll target. We just set the nll target to 0 in this case.
                            nll_per_snt[batch_idx].append(0)
                        else:
                            nll_per_snt[batch_idx].append(nll_per_token[batch_idx, relevant_idxs].mean())
                    # turn into tensor that is padded to num_steps
                    nll_per_snt[batch_idx] = torch.tensor(nll_per_snt[batch_idx][:num_steps] + [0] * max(0, num_steps - len(nll_per_snt[batch_idx]))).to(nll_per_token.device)
                nll_per_snt = torch.stack(nll_per_snt)
                target_reward = -nll_per_snt

                return target_reward

            raise NotImplementedError("Check if target_codes is right format for below (because I changed it to being compressed now")
            context_and_target_codes = torch.cat([context_codes, target_codes], dim=1)
            target_reward = apply_lm_with_codes(context_and_target_codes)
            if self.hparams.mz_reward_baseline is not None:
                if self.hparams.mz_reward_baseline == "nocc":
                    baseline_reward = apply_lm_with_codes(None)
                elif self.hparams.mz_reward_baseline == "random":
                    rand_codes = torch.randint_like(target_codes, low=0, high=self.lm.hparams.codebook.shape[0])
                    random_target_codes = torch.cat([context_codes, rand_codes], dim=1)
                    baseline_reward = apply_lm_with_codes(random_target_codes)
                else:
                    raise ValueError(f"Invalid value for mz_reward_baseline: {self.hparams.mz_reward_baseline}")
                
                target_reward = target_reward - baseline_reward

            # target value at step i = discounted sum of target reward from step i onwards.
            target_value = []
            for i in range(num_steps):
                target_value.append((target_reward[:, i:] * self.hparams.discount ** torch.arange(target_reward.shape[1] - i).to(target_reward.device)).mean(dim=1))
            target_value = torch.stack(target_value, dim=1)

            # loss calculation
            reward_loss_function = scalar_loss if self.hparams.mz_reward_loss_fn == "mse" else bce_loss
            value_loss_function = scalar_loss if self.hparams.mz_value_loss_fn == "mse" else bce_loss

            reward_loss  = reward_loss_function(pred_reward, target_reward, reduction='none')
            #print(f"Pred Reward: {pred_reward}\nTarget Reward: {target_reward}\nLoss: {reward_loss}")
            value_loss = value_loss_function(pred_value, target_value, reduction='none')
            # ce_loss = CrossEntropyLoss(ignore_index=-1)
            # policy_loss = CrossEntropyLoss(ignore_index=-1)(pred_policy.view(-1, pred_policy.shape[-1]), target_actions.view(-1).to(torch.long))

            total_loss = self.hparams.mz_reward_loss_lambda * reward_loss + \
                         self.hparams.mz_value_loss_lambda * value_loss + \
                         self.hparams.mz_policy_loss_lambda * policy_loss


            # following muzero pseudocode (https://arxiv.org/src/1911.08265v2/anc/pseudocode.py), the gradient is scaled by 1/(num_unroll_steps) to ensure the total gradient has a similar magnitude irrespective of the number of unroll steps.
            # contrary to the og muzero, different batch elements might have different unroll steps (depending on how many sentences fit in that particular batch element)
            if not self.hparams.mz_no_gradient_scaling:
                gradient_scale = 1.0 / torch.tensor([len(snt_idxs_compressed[i]) for i in range(len(snt_idxs_compressed))]).to(
                    torch.float32).to(pred_policy.device)
                gradient_scale = gradient_scale[:, None]
                loss = scale_gradient(total_loss, gradient_scale)
            else:
                loss = total_loss
            
            loss = (loss * mask.to(loss.device)).sum() / mask.sum()
            next_actions = None
        self.log_stuff(loss, mask, num_steps, oracle_actions, pred_policy, split, policy_loss, reward_loss, value_loss, next_actions)

        return loss

    def get_target_codes(self, codes_between_tokens, oracle_codes, snt_idxs):
        if self.hparams.nonoracle_fraction != 0.0:
            if codes_between_tokens:
                raise NotImplementedError("TODO implement cc_type=insert for nonoracle_fraction != 0.0")
            # use noisy codes
            '''
            if batch size = 2, first batch el has 3 sentences (lengths 2, 2, 1 (cut-off)) and second batch el has 2 sentences (lengths 3 (continues cut-off), 2)
            - snt_idxs = [[51, 51, 52, 52, 53], 
                          [53, 53, 53, 54, 54]]
            - oracle_codes could be: 
                        [[8, 8, 2, 2, 19],
                        [19, 19, 19, 2, 2]] (where by accident sentence 52 and 54 were assigned the same code)
            - unique_snt_idxs = [51, 52, 53, 54]
            - oracle_code_source: [8, 2, 19, 2] (found by looking up the code for each unique_snt_idx in oracle_codes)
            '''
            unique_snt_idxs, inverse_indices = torch.unique(snt_idxs, return_inverse=True)
            random_code_source = torch.randint_like(unique_snt_idxs, low=0, high=self.hparams.cluster_count)
            oracle_code_source = torch.stack([oracle_codes[snt_idxs == snt_idx][0] for snt_idx in unique_snt_idxs])
            target_code_source = torch.where(
                torch.rand_like(random_code_source.float()) < self.hparams.nonoracle_fraction, random_code_source,
                oracle_code_source)
            target_codes = target_code_source[inverse_indices]
        else:
            target_codes = oracle_codes
        return target_codes

    def get_actions_and_mask(self, num_steps, target_codes=None, oracle_codes=None, snt_idxs=None, snt_idxs_compressed=None, input_ids_and_codes=None, code_mask=None,split=None):
        if snt_idxs is not None:
            batch_size = oracle_codes.shape[0]
            # Get codes for each sentence in chunk by using snt_idxs
            # target_codes_compressed = [[target_codes[i][list(snt_idxs[i]).index(snt_idx)] for snt_idx in snt_idxs_compressed[i]] for i in range(len(snt_idxs_compressed))]
            # target_codes_compressed = []
            # oracle_codes_compressed = []
            # if self.hparams.soft_planner_targets:
            #     target_probs = []
            # for i in range(batch_size):
            #     target_codes_compressed.append([])
            #     oracle_codes_compressed.append([])
            #     if self.hparams.soft_planner_targets:
            #         target_probs.append([])
            #     for snt_idx in snt_idxs_compressed[i]:
            #         j = list(snt_idxs[i]).index(snt_idx)
            #         target_codes_compressed[i].append(target_codes[i][j])
            #         oracle_codes_compressed[i].append(oracle_codes[i][j])
            #         if self.hparams.soft_planner_targets:
            #             emb = self.dataset.split2sent_embs[split][snt_idx]
            #             kmeans_scores = self.dataset.kmeans.transform(emb[None])
            #             kmeans_probs = F.softmax(torch.tensor(kmeans_scores).to(self.device))
            #             target_probs[i].append(kmeans_probs)
            def get_compressed(codes=None):
                compressed = [[] for _ in range(batch_size)]

                # Precompute indices
                snt_idx_map = [dict((snt.item(), idx) for idx, snt in enumerate(snt_idxs[i])) for i in range(batch_size)]

                if codes is None:
                    unique_snt_idxs = {i for l in snt_idxs_compressed for i in l}
                    all_kmeans_probs = self.dataset.get_kmeans_probs(unique_snt_idxs, split)
                    sntidx2localidx = {snt_idx: idx for idx, snt_idx in enumerate(unique_snt_idxs)}

                for i in range(batch_size):
                    for snt_idx in snt_idxs_compressed[i]:
                        if codes is not None:
                            # j = list(snt_idxs[i]).index(snt_idx)
                            j = snt_idx_map[i].get(snt_idx)
                            compressed[i].append(codes[i][j])
                        else:
                            # compressed[i].append(self.dataset.get_kmeans_probs(snt_idx,split))
                            compressed[i].append(all_kmeans_probs[sntidx2localidx[snt_idx]])
                return compressed
            oracle_codes_compressed = get_compressed(oracle_codes)
            targets_compressed = get_compressed(target_codes if not self.hparams.soft_planner_targets else None)
            device = oracle_codes.device
        else:
            assert not self.hparams.soft_planner_targets, "Soft planner targets not implemented for code_between_tokens"
            # case code_between_tokens
            oracle_codes_compressed = [[el for el in input_ids_and_codes[i][code_mask[i]]] for i in range(len(input_ids_and_codes))]
            target_codes_compressed = oracle_codes_compressed # No support yet for nonoracle_fraction != 0.0 with code_between_tokens
            batch_size = input_ids_and_codes.shape[0]
            device = input_ids_and_codes.device
        # Get oracle codes by padding ..._compressed to num_steps
        if self.hparams.soft_planner_targets:
            # Replaced the assert below with a warning, as I think it might differ due to acceptable small randomness somewhere

            # assert [[el.argmax().item() for el in row] for row in targets_compressed] == [[el.item() for el in row] for row in oracle_codes_compressed], \
            # f"Expected target codes to be the same as oracle codes for soft planner targets, but got {[[el.argmax().item() for el in row] for row in targets_compressed]} and {[[el.item() for el in row] for row in oracle_codes_compressed]}"
            if (on_the_fly:=[[el.argmax().item() for el in row] for row in targets_compressed]) !=( precomputed:=[[el.item() for el in row] for row in oracle_codes_compressed]):
                print("WARNING: on-the-fly argmax not equal to oracle code")
                print(on_the_fly)
                print(precomputed)
                print(f"Number of differing codes: {sum([1 for i in range(len(on_the_fly)) for j in range(len(on_the_fly[i])) if on_the_fly[i][j] != precomputed[i][j]])}")
                print("Difference in score for differing codes:")
                for i in range(len(on_the_fly)):
                    for j in range(len(on_the_fly[i])):
                        if on_the_fly[i][j] != precomputed[i][j]:
                            print(f"argmax score: {targets_compressed[i][j][on_the_fly[i][j]]}, oracle score: {targets_compressed[i][j][precomputed[i][j]]}")


        # targets = []
        # oracle_actions = []
        # mask = []
        # for batch_idx in range(batch_size):
        #     target_actions_batch = targets_compressed[batch_idx][:num_steps]
        #     oracle_actions_batch = oracle_codes_compressed[batch_idx][:num_steps]
        #     mask_batch = [1] * len(target_actions_batch)
        #     if len(target_actions_batch) < num_steps:
        #         padding_length = num_steps - len(target_actions_batch)
        #         target_actions_batch += [0] * padding_length
        #         oracle_actions_batch += [0] * padding_length
        #         mask_batch += [0] * padding_length
        #
        #     targets.append(target_actions_batch)
        #     oracle_actions.append(oracle_actions_batch)
        #     mask.append(mask_batch)
        # targets = torch.tensor(targets).to(device)
        # oracle_actions = torch.tensor(oracle_actions).to(device)
        # mask = torch.tensor(mask).to(device)
        def get_num_steps(lst, mask=False):
            result = []
            for batch_idx in range(batch_size):
                lst_batch = lst[batch_idx][:num_steps] if not mask else [1] * len(lst[batch_idx][:num_steps])
                if len(lst_batch) < num_steps:
                    padding_length = num_steps - len(lst_batch)
                    lst_batch += [0] * padding_length
                result.append(lst_batch)

            if type(result[0][0]) == np.ndarray:
                try:
                    result = np.array(result)
                except:
                    pass

            return torch.tensor(result).to(device)

        oracle_actions = get_num_steps(oracle_codes_compressed)
        targets = get_num_steps(targets_compressed)
        mask = get_num_steps(targets_compressed, mask=True)

        return mask, oracle_actions, targets

    def _log_steps(self, loss, mask, num_steps, oracle_actions, pred_policy, split, policy_loss, reward_loss,
                  value_loss,next_actions, prefix):
        self.log(f'{prefix}{split}_policy_loss', (policy_loss * mask.to(policy_loss.device)).sum() / mask.sum())
        filtered_actions = torch.where(mask == 1, oracle_actions, -1)
        avg_rank = (pred_policy.argsort(descending=True) == filtered_actions.unsqueeze(-1)).nonzero(as_tuple=True)[
            -1].float().mean()
        self.log(f'{prefix}{split}_avg_rank_of_{pred_policy.shape[-1]}', avg_rank)

        if len(policy_loss.shape) == 1:
            policy_loss = policy_loss.unsqueeze(-1)
        if num_steps > 1:
            for step in range(num_steps):
                if reward_loss is not None:
                    self.log(f'per_step/{prefix}{split}_reward_loss_{step + 1}',
                                (reward_loss[:, step] * mask[:, step]).sum() / mask[:, step].sum())
                if value_loss is not None:
                    self.log(f'per_step/{prefix}{split}_value_loss_{step + 1}',
                            (value_loss[:, step] * mask[:, step]).sum() / mask[:, step].sum())
                if not self.hparams.soft_planner_targets:
                    self.log(f'per_step/{prefix}{split}_policy_loss_{step + 1}',
                            (policy_loss[:, step] * mask[:, step]).sum() / mask[:, step].sum())
                # avg_rank per step
                filtered_actions = torch.where(mask[:, step] == 1, oracle_actions[:, step], -1)
                rank = (pred_policy[:, step].argsort(descending=True) == filtered_actions.unsqueeze(-1)).nonzero(
                    as_tuple=True)[-1]
                avg_rank = rank.float().mean()
                avg_accuracy = (rank == 0).float().mean()
                self.log(f'per_step/{prefix}{split}_avg_rank_of_{pred_policy.shape[-1]}_{step + 1}', avg_rank)
                self.log(f'per_step/{prefix}{split}_avg_accuracy_{step + 1}', avg_accuracy)


    def log_stuff(self, loss, mask, num_steps, oracle_actions, pred_policy, split, policy_loss, reward_loss,
                  value_loss,next_actions):
        prefix = "mzpt/"

        self.log(f'{prefix}{split}_loss', loss, prog_bar=True)
        if self.hparams.only_policy_head:
            assert reward_loss is None and value_loss is None

            if is_embedding_space_prediction(self.hparams.mz_policy_loss_fn):
                cb_device = "cpu" if self.hparams.no_cluster else self.device
                pred_policy = get_regression_policy_logits(pred_policy, self.policy_loss_fn.codebook, self.policy_loss_fn.codebook_transform, self.hparams.mz_policy_loss_fn_distance_function, device=cb_device)


            self._log_steps(loss, mask, num_steps, oracle_actions, pred_policy, split, policy_loss, reward_loss,
                  value_loss,next_actions, prefix)

        else:
            assert next_actions is None
            assert self.hparams.mz_policy_loss_fn == "ce", "Regression loss only implemented for only-policy-head."
            self.log(f'{prefix}{split}_reward_loss', (reward_loss * mask.to(reward_loss.device)).sum() / mask.sum(), on_epoch=True)
            self.log(f'{prefix}{split}_value_loss', (value_loss * mask.to(value_loss.device)).sum() / mask.sum())
            # avg_rank = (pred_policy.argsort(descending=True) == oracle_actions.unsqueeze(-1)).nonzero(as_tuple=True)[1].float().mean()

            self._log_steps(loss, mask, num_steps, oracle_actions, pred_policy, split, policy_loss, reward_loss,
                  value_loss,next_actions, prefix)

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        # if self.model_h.tune == 0: # No need to save the base model if it's frozen
        if self.model_h.freeze:
            for k in list(checkpoint['state_dict']):
                if k.startswith('model_h.model'):
                    del checkpoint['state_dict'][k]
        # also lm should not be saved (because we might want to combine a different lm with muzero)
        for k in list(checkpoint['state_dict']):
            if k.startswith('lm'):
                del checkpoint['state_dict'][k]

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        # if self.model_h.tune == 0:
        if self.model_h.freeze:
            if self.hparams.mz_multi_vector_representation:
                base_model = SentenceTransformer(self.hparams.mz_h_sent_model, cache_folder=self.hparams.sentence_transformer_cache)
            else:
                if self.hparams.mz_h_model_name is not None:
                    model_name = self.hparams.mz_h_model_name
                else:
                    model_name = self.hparams.base_model_name
                if "sentence-transformers" in model_name or (model_name in SHORT2FULL_EMBEDDER_NAME):
                    if model_name in SHORT2FULL_EMBEDDER_NAME:
                        model_name = SHORT2FULL_EMBEDDER_NAME[model_name]
                    base_model = SentenceTransformer(model_name, cache_folder=self.hparams.sentence_transformer_cache)
                else:
                    base_model = AutoModelForCausalLM.from_pretrained(self.hparams.base_model_name, trust_remote_code=False)
            checkpoint['state_dict'].update({f'model_h.model.{k}': v for k, v in base_model.state_dict().items()})

        # Backwards compatibility
        for w_or_b in ['weight', 'bias']:
            newkey_g = f'model_g.common_layers.0.layer.0.{w_or_b}'
            newkey_f = f'model_f.model.0.layer.0.{w_or_b}'

            oldkeys_g = [f'model_g.common_layers.0.{w_or_b}', f'model_g.linear.{w_or_b}']
            oldkeys_f = [f'model_f.model.0.{w_or_b}', f'model_f.linear.{w_or_b}']

            for newkey, oldkeys in [(newkey_g, oldkeys_g), (newkey_f, oldkeys_f)]:
                if newkey not in checkpoint['state_dict']:
                    for oldkey in oldkeys:
                        if oldkey in checkpoint['state_dict']:
                            checkpoint['state_dict'][newkey] = checkpoint['state_dict'].pop(oldkey)
                            print(f"Renamed {oldkey} to {newkey}")
                            break

        # Backwards compatibility: ignore lm keys
        for k in list(checkpoint['state_dict']):
            if k.startswith('lm'):
                del checkpoint['state_dict'][k]
        # Also backwards compatibility: ignore entries for lm params in optimizer
        if len(list(self.parameters())) < len(checkpoint['optimizer_states'][0]['param_groups'][0]['params']):
            print("WARNING: DOING SOMETHING VERY HACKY WHICH IS ONLY RELEVANT IF TRYING TO RESUME FROM OLD CHECKPOINT IN WHICH LM PARAMS WERE STILL PART OF MZPT OPTIMIZER")
            checkpoint['optimizer_states'][0]['param_groups'][0]['params'] = checkpoint['optimizer_states'][0]['param_groups'][0]['params'][:len(list(self.parameters()))]

    def configure_optimizers(self):
        #exclude self.lm from parameters if self.lm is present
        parameters_without_lm = [p for (n,p) in self.named_parameters() if not n.startswith('lm.')]
        return torch.optim.Adam(parameters_without_lm, lr=self.hparams.mz_lr, weight_decay=self.hparams.mz_weight_decay)


    def on_train_epoch_start(self):
        if hasattr(self.model_g, 'action_embedder'):
            if self.current_epoch < self.hparams.action_embedder_frozen_epochs:
                self.model_g.action_embedder.requires_grad_(False)
            else:
                self.model_g.action_embedder.requires_grad_(True)

    def compute_codes_or_scores_for_articles(self, articles, bos_id, scores_iso_codes=False):
        all_context_ids, all_context_masks = [], []
        c_or_s = 'scores' if scores_iso_codes else 'codes'
        artidx2bound = {}
        prev_bound = 0
        for artidx, article in enumerate(tqdm(articles, desc="Gathering context_ids and context_masks for articles")):
            tokenized_sentences = article['tokenized_sentences']
            art_context_ids = torch.tensor([42]*(self.hparams.max_seq_len - 1) + [bos_id]).unsqueeze(0) # This is left padding, but in this case no problem because we convert to text anyway and ignore the padding
            art_context_mask = torch.tensor([0]*(self.hparams.max_seq_len - 1) + [1]).unsqueeze(0)
            for s in tokenized_sentences[:-1]:
                s = torch.tensor(s)
                new_context_ids = torch.cat([art_context_ids[-1][None], s[None]], dim=1)[:,-self.hparams.max_seq_len:]
                new_context_mask = torch.cat([art_context_mask[-1][None], torch.ones_like(s)[None]], dim=1)[:,-self.hparams.max_seq_len:]
                art_context_ids = torch.cat([art_context_ids, new_context_ids], dim=0)
                art_context_mask = torch.cat([art_context_mask, new_context_mask], dim=0)
            all_context_ids.append(art_context_ids)
            all_context_masks.append(art_context_mask)
            artidx2bound[artidx] = (prev_bound, prev_bound + len(tokenized_sentences))
            prev_bound += len(tokenized_sentences)
        # stack list into tensor
        all_context_ids = torch.cat(all_context_ids, dim=0)
        all_context_masks = torch.cat(all_context_masks, dim=0)


        codes_or_scores = []
        BS = 128
        with torch.no_grad():
            for i in tqdm(range(0, all_context_ids.shape[0], BS), desc=f"Computing planning {c_or_s} for articles"):
                representation = self.model_h(all_context_ids[i:i+BS].to('cuda'), all_context_masks[i:i+BS].to('cuda'))
                codes_or_scores.extend(self._get_planner_code_or_scores(representation, scores_iso_codes))

        result = []
        for artidx, article in tqdm(enumerate(articles), desc="Gathering planning codes in list"):
            planner_codes_or_scores = codes_or_scores[artidx2bound[artidx][0]:artidx2bound[artidx][1]]
            assert len(planner_codes_or_scores) == len(article['codes'])
            if scores_iso_codes:
                planner_codes_or_scores = np.stack(planner_codes_or_scores, axis=0)
            # article[f'planner_{"scores" if scores_iso_codes else "codes"}'] = planner_codes_or_scores
            result.append(planner_codes_or_scores)
        return result

    def _get_planner_code_or_scores(self, representation, scores_iso_codes=False):

        if self.hparams.eval_with_path_search:
            assert not scores_iso_codes
            code = muzero_beamsearch_batched(representation, self.model_f, self, self.hparams)
            #code2 = muzero_beamsearch_batched_vectorized(representation, self.model_f, self, self.hparams)

            #assert torch.equal(code,code2)
        else:
            logits = self.model_f(representation)
            if not self.hparams.only_policy_head:
                logits = logits[0]
            if scores_iso_codes:
                return tn(logits)
            code = logits.argmax(dim=-1).cpu().tolist()
        return code



    def get_greedy_logits(self, input_ids, attention_mask, cb=None):
        state_rep = self.model_h(input_ids, attention_mask)
        model_f_output = self.model_f(state_rep)
        policy_logits = model_f_output
        if not self.policy_only:
            policy_logits, _ = policy_logits

        if is_embedding_space_prediction(self.hparams.mz_policy_loss_fn):
            # not implemented for when we have a codebook transform
            assert self.hparams.mz_regression_loss_transform is None, "Not implemented for when we have a codebook transform"
            policy_logits = get_regression_policy_logits(policy_logits, cb, None,
                                                         self.hparams.mz_policy_loss_fn_distance_function, device=cb.device)

        return policy_logits

    def get_beamsearch_code(self, input_ids, attention_mask):
        state_rep = self.model_h(input_ids, attention_mask)
        search_code = muzero_beamsearch(state_rep, self.model_f, self, self.hparams)
        return search_code.long()

    def get_mcts_logits(self, input_ids, attention_mask):

        # Lazy importing b/c at the moment we're not doing mcts yet. Untested though :P
        def import_jax_stuff():
            global jax, mctx, jnp, t2j, j2t  # Access global to modify
            import jax
            import mctx
            from jax import numpy as jnp
            from torch2jax import t2j, j2t
            return jax, mctx, jnp, t2j, j2t

        if not all([k in globals() for k in ['jax', 'mctx', 'jnp', 't2j', 'j2t']]):
            jax, mctx, jnp, t2j, j2t = import_jax_stuff()

        if not hasattr(self, 'jax_model_f'):
            self.jax_model_f = t2j(self.model_f)
            self.state_dict_f = {k: t2j(v) for k, v in self.model_f.named_parameters()}
            self.jax_model_g = t2j(self.model_g)
            self.state_dict_g = {k: t2j(v) for k, v in self.model_g.named_parameters()}

        def recurrent_fn(params, rng_key, action, embedding):
            next_embedding, reward = self.jax_model_g((embedding, action), self.state_dict_g)
            prior_logits, value = self.jax_model_f(next_embedding, self.state_dict_f)
            recurrent_fn_output = mctx.RecurrentFnOutput(
                reward=reward,
                discount=jnp.ones_like(reward) * self.hparams.discount,
                prior_logits=prior_logits,
                value=value)
            return recurrent_fn_output, next_embedding

        state_rep = self.model_h(input_ids, attention_mask)
        if self.hparams.cc_type == 'insert':
            raise NotImplementedError("TODO implement for non-eval_only_policy_head")
        state_rep = t2j(state_rep)
        # policy_logits, value = jax_model_f(state_rep, state_dict_f)
        model_f_output = self.jax_model_f(state_rep, self.state_dict_f)
        policy_logits, value = model_f_output
        root = mctx.RootFnOutput(prior_logits=policy_logits, value=value, embedding=state_rep)
        policy_output = mctx.gumbel_muzero_policy(params=(), rng_key=jax.random.PRNGKey(self.hparams.mctx_seed), root=root,
                                                  recurrent_fn=recurrent_fn, num_simulations=self.hparams.num_simulations,
                                                  max_depth=self.hparams.max_depth)
        mcts_logits = j2t(policy_output.action_weights)
        return mcts_logits, policy_output

    def is_sentence_transformer_param(self, name):
        return 'model_h.model' in name

if __name__ == '__main__':
    mz = MuZero()
