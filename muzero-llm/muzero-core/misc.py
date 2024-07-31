import torch
import torch.nn.functional as F
from collections import Counter
from collections import defaultdict
import math

def muzero_beamsearch(state_rep, model_f, planner, args):
    beam_paths = [(state_rep, 0, [])]  # Initialize beam with state_rep, score 0, and empty index sequence
    for i in range(args.mz_max_timesteps):
        all_candidates = []
        for path, score, indices in beam_paths:
            if args.only_policy_head:
                policy_logits = model_f(path)
            else:
                policy_logits, _ = model_f(path)
            
            if i == 0:
                greedy_policy = policy_logits

            device = policy_logits.device
            # Convert logits to log probabilities
            log_probs = F.log_softmax(policy_logits, dim=-1)
            top_k_log_probs, top_k_indices = torch.topk(log_probs, args.eval_mz_search_K)

            # Prepare for batched dynamics function call
            batch_size, num_actions = top_k_log_probs.shape
            if isinstance(path, tuple):
                expanded_paths = tuple(torch.cat([p for _ in range(num_actions)], dim=0) for p in path)
            else:
                expanded_paths = torch.cat([path for _ in range(num_actions)], dim=0)
            # vectorized NN call
            dynamics, rewards = planner.single_step(expanded_paths, top_k_indices.view(-1))
            
            # Split each tensor in the dynamics tuple if dynamics is a tuple (consisting of (state, length))
            if isinstance(dynamics, tuple):
                dynamics = list(zip(*(d.split(batch_size, dim=0) for d in dynamics)))
            else:
                dynamics = dynamics.split(batch_size, dim=0)

            if args.eval_mz_search_score == "policy":

                for d_ind, (log_prob, index) in enumerate(zip(top_k_log_probs.squeeze(), top_k_indices.squeeze())):
                    # Update path score with log probability
                    candidate_score = score + log_prob.item()  # Add log probabilities
                    updated_indices = indices + [index.item()]  # Append the current index to the sequence
                    all_candidates.append((dynamics[d_ind], candidate_score, updated_indices))
            
            elif args.eval_mz_search_score == "reward":

                rewards = rewards.split(batch_size, dim=0)
                for r_ind, (reward, index) in enumerate(zip(rewards, top_k_indices.squeeze())):
                    # Use reward as score
                    candidate_score = score + reward.item()  # Add reward
                    updated_indices = indices + [index.item()]  # Append the current index to the sequence
                    all_candidates.append((dynamics[r_ind], candidate_score, updated_indices))
            else:
                raise ValueError("Invalid eval_mz_search_score specified")

        # Sort all candidates by score in descending order and select top K, keeping track of the sequence of indices
        beam_paths = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:args.eval_mz_search_K]

        #print("Iteration:", i)
        #print("Candidate Indices, Scores, and Greedy Policy Scores:")
        #for path in beam_paths:
        #    print("Indices:", path[2], "Score:", path[1], "Greedy Policy Score:", greedy_policy[0, path[2][0]].item())

    # Select action based on the eval_mz_search_selection flag
    if args.eval_mz_search_selection == "top_path":
        selected_action, _ = get_top_path_action(beam_paths)
        
    if args.eval_mz_search_selection == "top_path_threshold":
        selected_action, top_path_value = get_top_path_action(beam_paths)
        if top_path_value > args.eval_mz_search_threshold:
            # Use argmax of the policy if value does not exceed threshold
            selected_action = torch.argmax(greedy_policy, dim=-1).item()

    elif args.eval_mz_search_selection == "most_visited":
        selected_action = get_most_visited(beam_paths)
    elif args.eval_mz_search_selection == "most_likely":
        selected_action = get_most_likely(beam_paths)
    elif args.eval_mz_search_selection == "most_likely_average":
        selected_action = get_most_likely_average(beam_paths)
    else:
        raise ValueError("Invalid selection method specified for eval_mz_search_selection")
    
    return torch.tensor([selected_action], device=device).float().view(1,-1)

def muzero_beamsearch_batched(representation, model_f, planner, args):
    all_codes = []
    if isinstance(representation, tuple):
        rep, lengths = representation
        for i in range(rep.shape[0]):
            single_rep = rep[i].unsqueeze(0)  # Add batch dimension
            length = lengths[i].unsqueeze(0)
            code = muzero_beamsearch((single_rep, length), model_f, planner, args)
            all_codes.append(code)
    else:
        for i in range(representation.shape[0]):
            single_rep = representation[i].unsqueeze(0)  # Add batch dimension
            code = muzero_beamsearch(single_rep, model_f, planner, args)
            all_codes.append(code)
    return torch.cat(all_codes, dim=0)

def muzero_beamsearch_batched_vectorized(representation, model_f, planner, args):
    if isinstance(representation, tuple):
        state_rep, _ = representation  # state_rep: [batch_size, seq_length, dim]
        batch_size = state_rep.shape[0]
        device = state_rep.device
        num_dims = state_rep.dim()
    else:
        batch_size = representation.shape[0]  # representation: [batch_size, seq_length, dim]
        device = representation.device
        num_dims = representation.dim()
    assert num_dims == 3, "Expected representation to have 3 dimensions [batch_size, seq_length, dim]"
    
    # Initialize beam with state_rep, score 0, and empty index sequence for each item in the batch
    initial_scores = torch.zeros(batch_size, args.eval_mz_search_K, device=device)  # initial_scores: [batch_size, K]
    initial_indices = torch.zeros(batch_size, args.eval_mz_search_K, 1, dtype=torch.long, device=device)  # initial_indices: [batch_size, K, 1]
    beam_paths = [(representation, initial_scores, initial_indices)]  # beam_paths: list of tuples
    
    for i in range(args.mz_max_timesteps):
        all_candidates = []
        for path, scores, indices_list in beam_paths:
            policy_logits, _ = model_f(path) if not args.only_policy_head else (model_f(path), None)  # policy_logits: [batch_size, num_actions]
            assert policy_logits.dim() == 2, "Expected policy_logits to have 2 dimensions [batch_size, num_actions]"
            
            if i == 0:
                greedy_policy = policy_logits  # greedy_policy: [batch_size, num_actions]

            device = policy_logits.device
            # Convert logits to log probabilities
            log_probs = F.log_softmax(policy_logits, dim=-1)  # log_probs: [batch_size, num_actions]
            top_k_log_probs, top_k_indices = torch.topk(log_probs, args.eval_mz_search_K, dim=-1)  # top_k_log_probs, top_k_indices: [batch_size, K]

            if isinstance(path, tuple):
                expanded_paths = tuple(p.repeat_interleave(args.eval_mz_search_K, dim=0) for p in path)
            else:
                expanded_paths = path.repeat_interleave(args.eval_mz_search_K, dim=0)  # expanded_paths: [batch_size*K, seq_length, dim]
            # vectorized NN call
            dynamics, rewards = planner.single_step(expanded_paths, top_k_indices.view(-1))  # dynamics: [batch_size*K, seq_length, dim], rewards: [batch_size*K]
            if isinstance(dynamics, tuple):
                dynamics = (dynamics[0].reshape(batch_size, args.eval_mz_search_K, -1), dynamics[1].reshape(batch_size, args.eval_mz_search_K))  # dynamics: tuple of [batch_size, K, seq_length, dim]
            else:
                dynamics = dynamics.view(batch_size, args.eval_mz_search_K, -1)  # dynamics: [batch_size, K, seq_length, dim]
            rewards = rewards.view(batch_size, args.eval_mz_search_K)  # rewards: [batch_size, K]
            
            print(scores.size())
            print(top_k_log_probs.size())
            # Update scores and indices for all candidates
            candidate_scores = scores + top_k_log_probs  # candidate_scores: [batch_size, K, 1]
            candidate_indices = torch.cat((indices_list, top_k_indices.unsqueeze(-1)), dim=2)  # candidate_indices: [batch_size, K, seq_length+1]

            # Append updated paths, scores, and indices
            all_candidates.append((dynamics, candidate_scores, candidate_indices))  # dynamics: [batch_size, K, seq_length, dim], candidate_scores: [batch_size, K, 1]

        # Process all_candidates while considering batch dimension
        # Handle the case when dynamics is a tuple
        if isinstance(all_candidates[0][0], tuple):
            dynamics_0, dynamics_1 = zip(*[d for d, _, _ in all_candidates])
            dynamics_0 = torch.stack(dynamics_0)  # [num_candidates, batch_size, K, seq_length, dim]
            dynamics_1 = torch.stack(dynamics_1)  # [num_candidates, batch_size, K, seq_length, dim]
            dynamics = (dynamics_0, dynamics_1)
        else:
            dynamics = torch.stack([d for d, _, _ in all_candidates])  # [num_candidates, batch_size, K, seq_length, dim]
        scores = torch.stack([s for _, s, _ in all_candidates])  # [num_candidates, batch_size, K]
        indices = torch.stack([i for _, _, i in all_candidates])  # [num_candidates, batch_size, K, seq_length+1]
        # Sort scores along the candidates dimension and select top K for each batch
        sorted_scores, sorted_indices = scores.sort(dim=0, descending=True)
        top_k_sorted_indices = sorted_indices[:args.eval_mz_search_K]

        # Gather the top K dynamics and indices based on sorted indices
        if isinstance(dynamics, tuple):
            top_k_dynamics = tuple(d.index_select(0, top_k_sorted_indices.view(-1)).view(-1, batch_size, args.eval_mz_search_K, d.size(-2), d.size(-1)) for d in dynamics)
        else:
            top_k_dynamics = dynamics.index_select(0, top_k_sorted_indices.view(-1)).view(-1, batch_size, args.eval_mz_search_K, dynamics.size(-2), dynamics.size(-1))
        top_k_indices = indices.index_select(0, top_k_sorted_indices.view(-1)).view(-1, batch_size, args.eval_mz_search_K, indices.size(-1))

        batched_all_candidates = [(top_k_dynamics[:, i], sorted_scores[:args.eval_mz_search_K, i], top_k_indices[:, i]) for i in range(batch_size)]

        # Reorganize the data to match the expected format
        beam_paths = list(zip(*batched_all_candidates))  # beam_paths: list of tuples
        beam_paths = [(torch.stack(d), torch.stack(s), torch.cat(i, dim=-1)) for d, s, i in beam_paths]  # d: [batch_size, K, seq_length, dim], s: [batch_size, K, 1], i: [batch_size, K, seq]

    # Select action based on the eval_mz_search_selection flag
    selected_actions = []
    for path, score, indices in beam_paths:
        if args.eval_mz_search_selection == "top_path":
            selected_actions.append(indices[:, 0])  # indices[:, 0]: [batch_size]
        elif args.eval_mz_search_selection == "top_path_threshold":
            top_path_indices = indices[:, 0]  # top_path_indices: [batch_size]
            top_path_scores = score[:, 0]  # top_path_scores: [batch_size]
            threshold_mask = top_path_scores > args.eval_mz_search_threshold
            selected_actions.append(torch.where(threshold_mask, torch.argmax(greedy_policy, dim=-1), top_path_indices))
        elif args.eval_mz_search_selection == "most_visited":
            selected_actions.append(get_most_visited(indices))  # indices: [batch_size, K, seq]
        elif args.eval_mz_search_selection == "most_likely":
            selected_actions.append(get_most_likely(indices))  # indices: [batch_size, K, seq]
        elif args.eval_mz_search_selection == "most_likely_average":
            selected_actions.append(get_most_likely_average(indices))  # indices: [batch_size, K, seq]
        else:
            raise ValueError("Invalid selection method specified for eval_mz_search_selection")
    
    return torch.stack(selected_actions).to(device).float().view(-1,1)  # selected_actions: [batch_size]
    
def get_top_path_action(beam_paths):
    # Return the first action of the top-scoring path as an integer. beam_paths are already sorted.
    return beam_paths[0][2][0], beam_paths[0][1]

def get_most_visited(beam_paths):
    # Extract the first action of each path
    first_actions = [path[2][0] for path in beam_paths if path[2]]
    # Count the occurrences of each action
    action_counts = Counter(first_actions)
    # Find the most common action
    most_common_action, _ = action_counts.most_common(1)[0]
    return most_common_action if action_counts else None

def get_most_likely(beam_paths):
    # Segment beam paths by the first action in each path
    action_to_probs = defaultdict(float)
    for path in beam_paths:
        first_action = path[2][0]  # Assuming path[2] is the list of actions
        path_score = path[1]  # Assuming path[1] is the score (log probability)
        action_to_probs[first_action] += math.exp(path_score)  # Convert log probability to probability

    # Find the action with the maximum cumulative probability
    most_likely_action = max(action_to_probs, key=action_to_probs.get)

    return most_likely_action

def get_most_likely_average(beam_paths):
    # Segment beam paths by the first action in each path and average their probabilities
    action_to_probs = defaultdict(list)
    for path in beam_paths:
        first_action = path[2][0]  # Assuming path[2] is the list of actions
        path_score = path[1]  # Assuming path[1] is the score (log probability)
        action_to_probs[first_action].append(math.exp(path_score))  # Convert log probability to probability and collect

    # Calculate the average probability for each action
    for action, probs in action_to_probs.items():
        action_to_probs[action] = sum(probs) / len(probs)

    # Find the action with the maximum average probability
    most_likely_action = max(action_to_probs, key=action_to_probs.get)

    return most_likely_action
