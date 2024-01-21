from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rome import repr_tools
from util import nethook

from .memit_hparams import MEMITHyperParams



def compute_z_balance(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    hparams: MEMITHyperParams,
    layer: int,
    context_templates: List[str],
    flip_loss: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the value (right) vector for the rank-1 update.
    Runs a simple optimization procedure.
    """

    # Get model parameters
    lm_w, ln_f = (
        nethook.get_parameter(model, f"{hparams.lm_head_module}.weight").T,
        nethook.get_module(model, hparams.ln_f_module),
    )
    try:
        lm_b = nethook.get_parameter(model, f"{hparams.lm_head_module}.bias")
    except LookupError as _:
        lm_b = next(model.parameters()).new_zeros(model.config.vocab_size)

    print("Computing right vector (v)")

    # Tokenize target into list of int token IDs
    rewriting_prompts = []
    target_new_strs = [' he', ' she']
    for target_new_str in target_new_strs:

        target_ids = tok(target_new_str, return_tensors="pt").to("cuda")[
            "input_ids"
        ][0]

        # Compile list of rewriting and KL x/y pairs
        rewriting_prompts.extend([
            context.format(request["prompt"]) + tok.decode(target_ids[:-1])
            for context_types in context_templates
            for context in context_types
        ])
    print(">> len(rewriting_prompts)", len(rewriting_prompts))
    print(">> rewriting_prompts[0]", rewriting_prompts[0])
    print(">> rewriting_prompts[len(rewriting_prompts) // 2]", rewriting_prompts[len(rewriting_prompts) // 2])

    kl_prompts = ["{} is a"]
    all_prompts = rewriting_prompts + kl_prompts

    input_tok = tok(
        [prompt.format(request["subject"]) for prompt in all_prompts],
        return_tensors="pt",
        padding=True,
    ).to("cuda")

    # Compute rewriting targets
    rewriting_targets = torch.tensor(-100, device="cuda").repeat(
        len(rewriting_prompts), *input_tok["input_ids"].shape[1:]
    )
    for i in range(len(rewriting_prompts)):
        target_new_str = target_new_strs[i // (len(rewriting_prompts) // 2)] # pick he/she based on 1st or 2nd half
        target_ids = tok(target_new_str, return_tensors="pt").to("cuda")[
            "input_ids"
        ][0]
        ex_len = input_tok["attention_mask"][i].sum()
        rewriting_targets[i, ex_len - len(target_ids) : ex_len] = target_ids

    # Compute indices of the tokens where the fact is looked up
    lookup_idxs = [
        find_fact_lookup_idx(
            prompt, request["subject"], tok, hparams.fact_token, verbose=(i == 0)
        )
        for i, prompt in enumerate(all_prompts)
    ]

    # Finalize rewrite and loss layers
    loss_layer = max(hparams.v_loss_layer, layer)
    print(f"Rewrite layer is {layer}")
    print(f"Tying optimization objective to {loss_layer}")

    # Set up an optimization over a latent vector that, when output at the
    # rewrite layer, i.e. hypothesized fact lookup location, will induce the
    # target token to be predicted at the final layer.
    delta = torch.zeros((model.config.n_embd,), requires_grad=True, device="cuda")
    target_init, kl_distr_init = None, None

    # Inserts new "delta" variable at the appropriate part of the computation
    def edit_output_fn(cur_out, cur_layer):
        nonlocal target_init

        if cur_layer == hparams.layer_module_tmp.format(layer):
            # Store initial value of the vector of interest
            if target_init is None:
                print("Recording initial value of v*")
                # Initial value is recorded for the clean sentence
                target_init = cur_out[0][0, lookup_idxs[0]].detach().clone()

            # Add intervened delta
            for i, idx in enumerate(lookup_idxs):
                cur_out[0][i, idx, :] += delta

        return cur_out

    # Optimizer
    opt = torch.optim.Adam([delta], lr=hparams.v_lr)
    nethook.set_requires_grad(False, model)

    # Execute optimization
    for it in range(hparams.v_num_grad_steps):
        opt.zero_grad()

        # Forward propagation
        with nethook.TraceDict(
            module=model,
            layers=[
                hparams.layer_module_tmp.format(loss_layer),
                hparams.layer_module_tmp.format(layer),
            ],
            retain_input=False,
            retain_output=True,
            edit_output=edit_output_fn,
        ) as tr:
            logits = model(**input_tok).logits

            # Compute distribution for KL divergence
            kl_logits = torch.stack(
                [
                    logits[i - len(kl_prompts), idx, :]
                    for i, idx in enumerate(lookup_idxs[-len(kl_prompts) :])
                ],
                dim=0,
            )
            kl_log_probs = torch.nn.functional.log_softmax(kl_logits, dim=1)
            if kl_distr_init is None:
                kl_distr_init = kl_log_probs.detach().clone()

        # Compute loss on rewriting targets
        nll_loss_dict = {}
        for i, target_new_str in enumerate(target_new_strs):
            print("computing loss for target_new_str", target_new_str) 
            full_repr = tr[hparams.layer_module_tmp.format(loss_layer)].output[0][
                (len(rewriting_prompts)//2)*(i) : (len(rewriting_prompts)//2)*(i+1)
            ]
            cur_rewriting_targets = rewriting_targets[(len(rewriting_prompts)//2)*(i) : (len(rewriting_prompts)//2)*(i+1)]
            log_probs = torch.log_softmax(ln_f(full_repr) @ lm_w + lm_b, dim=2)
            loss = torch.gather(
                log_probs,
                2,
                torch.where(cur_rewriting_targets != -100, cur_rewriting_targets, 0).unsqueeze(2),
            ).squeeze(2)
            mask = (cur_rewriting_targets != -100).float()

            # Aggregate total losses
            nll_loss_each = -(loss * mask).sum(1) / target_ids.size(0)
            nll_loss_dict[target_new_str] = nll_loss_each.mean()

        # make the losses close together
        nll_loss = torch.abs(nll_loss_dict[target_new_strs[0]] - nll_loss_dict[target_new_strs[1]])
        kl_loss = hparams.kl_factor * torch.nn.functional.kl_div(
            kl_distr_init, kl_log_probs, log_target=True, reduction="batchmean"
        )
        weight_decay = hparams.v_weight_decay * (
            torch.norm(delta) / torch.norm(target_init) ** 2
        )
        # weight_decay = hparams.v_weight_decay * torch.norm(delta) ** 2
        loss = nll_loss + kl_loss + weight_decay
        print(
            f"loss {np.round(loss.item(), 3)} = {np.round(nll_loss.item(), 3)} + {np.round(kl_loss.item(), 3)} + {np.round(weight_decay.item(), 3)} "
            # f"avg prob of [{request['target_new']['str']}] "
            # f"{torch.exp(-nll_loss_each).mean().item()}"
        )
        if loss < 5e-2:
            break

        if it == hparams.v_num_grad_steps - 1:
            break

        # Backpropagate
        loss.backward()
        opt.step()

        # Project within L2 ball
        max_norm = hparams.clamp_norm_factor * target_init.norm()
        if delta.norm() > max_norm:
            with torch.no_grad():
                delta[...] = delta * max_norm / delta.norm()

    target = target_init + delta
    print(
        f"Init norm {target_init.norm()} | Delta norm {delta.norm()} | Target norm {target.norm()}"
    )

    return target

def compute_z(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    hparams: MEMITHyperParams,
    layer: int,
    context_templates: List[str],
    flip_loss: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the value (right) vector for the rank-1 update.
    Runs a simple optimization procedure.
    """

    # Get model parameters
    lm_w, ln_f = (
        nethook.get_parameter(model, f"{hparams.lm_head_module}.weight").T,
        nethook.get_module(model, hparams.ln_f_module),
    )
    try:
        lm_b = nethook.get_parameter(model, f"{hparams.lm_head_module}.bias")
    except LookupError as _:
        lm_b = next(model.parameters()).new_zeros(model.config.vocab_size)

    print("Computing right vector (v)")

    # Tokenize target into list of int token IDs
    target_ids = tok(request["target_new"]["str"], return_tensors="pt").to("cuda")[
        "input_ids"
    ][0]

    # Compile list of rewriting and KL x/y pairs
    rewriting_prompts, kl_prompts = [
        context.format(request["prompt"]) + tok.decode(target_ids[:-1])
        for context_types in context_templates
        for context in context_types
    ], ["{} is a"]
    all_prompts = rewriting_prompts + kl_prompts

    input_tok = tok(
        [prompt.format(request["subject"]) for prompt in all_prompts],
        return_tensors="pt",
        padding=True,
    ).to("cuda")

    # Compute rewriting targets
    rewriting_targets = torch.tensor(-100, device="cuda").repeat(
        len(rewriting_prompts), *input_tok["input_ids"].shape[1:]
    )
    for i in range(len(rewriting_prompts)):
        ex_len = input_tok["attention_mask"][i].sum()
        rewriting_targets[i, ex_len - len(target_ids) : ex_len] = target_ids

    # Compute indices of the tokens where the fact is looked up
    lookup_idxs = [
        find_fact_lookup_idx(
            prompt, request["subject"], tok, hparams.fact_token, verbose=(i == 0)
        )
        for i, prompt in enumerate(all_prompts)
    ]

    # Finalize rewrite and loss layers
    loss_layer = max(hparams.v_loss_layer, layer)
    print(f"Rewrite layer is {layer}")
    print(f"Tying optimization objective to {loss_layer}")

    # Set up an optimization over a latent vector that, when output at the
    # rewrite layer, i.e. hypothesized fact lookup location, will induce the
    # target token to be predicted at the final layer.
    delta = torch.zeros((model.config.n_embd,), requires_grad=True, device="cuda")
    target_init, kl_distr_init = None, None

    # Inserts new "delta" variable at the appropriate part of the computation
    def edit_output_fn(cur_out, cur_layer):
        nonlocal target_init

        if cur_layer == hparams.layer_module_tmp.format(layer):
            # Store initial value of the vector of interest
            if target_init is None:
                print("Recording initial value of v*")
                # Initial value is recorded for the clean sentence
                target_init = cur_out[0][0, lookup_idxs[0]].detach().clone()

            # Add intervened delta
            for i, idx in enumerate(lookup_idxs):
                cur_out[0][i, idx, :] += delta

        return cur_out

    # Optimizer
    opt = torch.optim.Adam([delta], lr=hparams.v_lr)
    nethook.set_requires_grad(False, model)

    # Execute optimization
    for it in range(hparams.v_num_grad_steps):
        opt.zero_grad()

        # Forward propagation
        with nethook.TraceDict(
            module=model,
            layers=[
                hparams.layer_module_tmp.format(loss_layer),
                hparams.layer_module_tmp.format(layer),
            ],
            retain_input=False,
            retain_output=True,
            edit_output=edit_output_fn,
        ) as tr:
            logits = model(**input_tok).logits

            # Compute distribution for KL divergence
            kl_logits = torch.stack(
                [
                    logits[i - len(kl_prompts), idx, :]
                    for i, idx in enumerate(lookup_idxs[-len(kl_prompts) :])
                ],
                dim=0,
            )
            kl_log_probs = torch.nn.functional.log_softmax(kl_logits, dim=1)
            if kl_distr_init is None:
                kl_distr_init = kl_log_probs.detach().clone()

        # Compute loss on rewriting targets
        full_repr = tr[hparams.layer_module_tmp.format(loss_layer)].output[0][
            : len(rewriting_prompts)
        ]
        log_probs = torch.log_softmax(ln_f(full_repr) @ lm_w + lm_b, dim=2)
        loss = torch.gather(
            log_probs,
            2,
            torch.where(rewriting_targets != -100, rewriting_targets, 0).unsqueeze(2),
        ).squeeze(2)
        mask = (rewriting_targets != -100).float()

        # Aggregate total losses
        nll_loss_each = -(loss * mask).sum(1) / target_ids.size(0)
        if flip_loss:
            print("flipping loss")
            nll_loss_each *= -1
        nll_loss = nll_loss_each.mean()
        kl_loss = hparams.kl_factor * torch.nn.functional.kl_div(
            kl_distr_init, kl_log_probs, log_target=True, reduction="batchmean"
        )
        weight_decay = hparams.v_weight_decay * (
            torch.norm(delta) / torch.norm(target_init) ** 2
        )
        # weight_decay = hparams.v_weight_decay * torch.norm(delta) ** 2
        loss = nll_loss + kl_loss + weight_decay
        if not flip_loss:
            print(
                f"loss {np.round(loss.item(), 3)} = {np.round(nll_loss.item(), 3)} + {np.round(kl_loss.item(), 3)} + {np.round(weight_decay.item(), 3)} "
                f"avg prob of [{request['target_new']['str']}] "
                f"{torch.exp(-nll_loss_each).mean().item()}"
            )
        else:
            print(
                f"loss {np.round(loss.item(), 3)} = {np.round(nll_loss.item(), 3)} + {np.round(kl_loss.item(), 3)} + {np.round(weight_decay.item(), 3)} "
                f"avg prob of [{request['target_new']['str']}] "
                f"{torch.exp(nll_loss_each).mean().item()}"
            )
        
        if not flip_loss and loss < 5e-2:
            break

        if flip_loss and loss < -50: # TODO: validate this stopping condition
            break

        if it == hparams.v_num_grad_steps - 1:
            break

        # Backpropagate
        loss.backward()
        opt.step()

        # Project within L2 ball
        max_norm = hparams.clamp_norm_factor * target_init.norm()
        if delta.norm() > max_norm:
            with torch.no_grad():
                delta[...] = delta * max_norm / delta.norm()

    target = target_init + delta
    print(
        f"Init norm {target_init.norm()} | Delta norm {delta.norm()} | Target norm {target.norm()}"
    )

    return target

def compute_z_with_anti(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    hparams: MEMITHyperParams,
    layer: int,
    context_templates: List[str],
    flip_loss: bool = False,
    debug_double_check: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the value (right) vector for the rank-1 update.
    Runs a simple optimization procedure.
    """

    # Get model parameters
    lm_w, ln_f = (
        nethook.get_parameter(model, f"{hparams.lm_head_module}.weight").T,
        nethook.get_module(model, hparams.ln_f_module),
    )
    try:
        lm_b = nethook.get_parameter(model, f"{hparams.lm_head_module}.bias")
    except LookupError as _:
        lm_b = next(model.parameters()).new_zeros(model.config.vocab_size)

    print("Computing right vector (v)")

    # Tokenize target into list of int token IDs
    target_ids = tok(request["target_new"]["str"], return_tensors="pt").to("cuda")["input_ids"][0]
    target_anti_ids = tok(request["target_anti"]["str"], return_tensors="pt").to("cuda")["input_ids"][0]

    # Compile list of rewriting and KL x/y pairs
    rewriting_prompts = [
        context.format(request["prompt"]) + tok.decode(target_ids[:-1])
        for context_types in context_templates
        for context in context_types
    ]
    rewriting_prompts_anti = [
        context.format(request["prompt"]) + tok.decode(target_ids[:-1])
        for context_types in context_templates
        for context in context_types
    ]
    assert len(rewriting_prompts) == len(rewriting_prompts_anti)
    
    kl_prompts = ["{} is a"] if (len(request["subject"].split()) > 1 or request["metadata"] == 'sg') else ["{} are a"]
    # print("DEBUG: using kl_prompts", kl_prompts, "for prompt", request["prompt"], request["subject"])
    all_prompts = rewriting_prompts + rewriting_prompts_anti + kl_prompts

    input_tok = tok(
        [prompt.format(request["subject"]) for prompt in all_prompts],
        return_tensors="pt",
        padding=True,
    ).to("cuda")

    # Compute rewriting targets
    rewriting_targets = torch.tensor(-100, device="cuda").repeat(
        len(rewriting_prompts + rewriting_prompts_anti), *input_tok["input_ids"].shape[1:]
    )
    for i in range(len(rewriting_prompts)):
        ex_len = input_tok["attention_mask"][i].sum()
        rewriting_targets[i, ex_len - len(target_ids) : ex_len] = target_ids
    for j in range(len(rewriting_prompts_anti)):
        i = j + len(rewriting_prompts)
        ex_len = input_tok["attention_mask"][i].sum()
        rewriting_targets[i, ex_len - len(target_anti_ids) : ex_len] = target_anti_ids

    # Compute indices of the tokens where the fact is looked up
    lookup_idxs = [
        find_fact_lookup_idx(
            prompt, request["subject"], tok, hparams.fact_token, verbose=(i == 0)
        )
        for i, prompt in enumerate(all_prompts)
    ]

    # Finalize rewrite and loss layers
    loss_layer = max(hparams.v_loss_layer, layer)
    print(f"Rewrite layer is {layer}")
    print(f"Tying optimization objective to {loss_layer}")

    # Set up an optimization over a latent vector that, when output at the
    # rewrite layer, i.e. hypothesized fact lookup location, will induce the
    # target token to be predicted at the final layer.
    delta = torch.zeros((model.config.n_embd,), requires_grad=True, device="cuda")
    target_init, kl_distr_init = None, None

    # Inserts new "delta" variable at the appropriate part of the computation
    def edit_output_fn(cur_out, cur_layer):
        nonlocal target_init

        if cur_layer == hparams.layer_module_tmp.format(layer):
            # Store initial value of the vector of interest
            if target_init is None:
                print("Recording initial value of v*")
                # Initial value is recorded for the clean sentence
                target_init = cur_out[0][0, lookup_idxs[0]].detach().clone()

            # Add intervened delta
            for i, idx in enumerate(lookup_idxs):
                cur_out[0][i, idx, :] += delta

        return cur_out

    # Optimizer
    opt = torch.optim.Adam([delta], lr=hparams.v_lr)
    nethook.set_requires_grad(False, model)

    # Execute optimization
    for it in range(hparams.v_num_grad_steps):
        opt.zero_grad()

        # Forward propagation
        with nethook.TraceDict(
            module=model,
            layers=[
                hparams.layer_module_tmp.format(loss_layer),
                hparams.layer_module_tmp.format(layer),
            ],
            retain_input=False,
            retain_output=True,
            edit_output=edit_output_fn,
        ) as tr:
            logits = model(**input_tok).logits

            # Compute distribution for KL divergence
            kl_logits = torch.stack(
                [
                    logits[i - len(kl_prompts), idx, :]
                    for i, idx in enumerate(lookup_idxs[-len(kl_prompts) :])
                ],
                dim=0,
            )
            kl_log_probs = torch.nn.functional.log_softmax(kl_logits, dim=1)
            if kl_distr_init is None:
                kl_distr_init = kl_log_probs.detach().clone()

        # Compute loss on rewriting targets
        full_repr = tr[hparams.layer_module_tmp.format(loss_layer)].output[0][
            : len(rewriting_prompts + rewriting_prompts_anti)
        ]
        # print("full_repr.shape", full_repr.shape)
        intermed_var = ln_f(full_repr) @ lm_w + lm_b
        # print("intermed_var.shape", intermed_var.shape)
        log_probs = torch.log_softmax(intermed_var, dim=2)
        # print("log_probs.shape", log_probs.shape)
        loss = torch.gather(
            log_probs,
            2,
            torch.where(rewriting_targets != -100, rewriting_targets, 0).unsqueeze(2),
        ).squeeze(2)
        # print("loss.shape", loss.shape)

        # TODO: check
        mask = (rewriting_targets != -100).float()
        # mask = (rewriting_targets != -100).float()
        # Aggregate total losses
        nll_loss_each = -(loss * mask).sum(1) / target_ids.size(0)
        if flip_loss:
            print("flipping loss")
            nll_loss_each *= -1
        nll_loss_target = nll_loss_each[:len(rewriting_prompts)].mean()
        nll_loss_anti = nll_loss_each[len(rewriting_prompts):len(rewriting_prompts)+len(rewriting_prompts_anti)].mean()
        # Loss is -log(p(A)/p(B)) = log(p(B)) - log(p(A)) = -log(p(A)) - -log(p(B))
        nll_loss = nll_loss_target - nll_loss_anti 
        # print("i", 0, "->", nll_loss_each[:10])
        # print("i", 1, "->", nll_loss_each[len(rewriting_prompts):len(rewriting_prompts)+10])

        if debug_double_check:
            # verify correctness
            nll_loss_dict = {}
            for i in range(2):
                # print("computing loss for target or anti-target:", i) 
                full_repr_redo = tr[hparams.layer_module_tmp.format(loss_layer)].output[0][
                    (len(rewriting_prompts))*(i) : (len(rewriting_prompts))*(i+1)
                ]
                assert ( torch.isclose(full_repr[(len(rewriting_prompts))*(i) : (len(rewriting_prompts))*(i+1)], full_repr_redo).all())
                cur_rewriting_targets = rewriting_targets[(len(rewriting_prompts))*(i) : (len(rewriting_prompts))*(i+1)]
                
                # intermed_var_redo = ln_f(full_repr_redo) @ lm_w + lm_b
                # print("DEBUG: intermed_var_redo", torch.isclose(
                #     intermed_var[(len(rewriting_prompts))*(i) : (len(rewriting_prompts))*(i+1)], 
                #     intermed_var_redo).all())
                # print(torch.max(torch.abs( intermed_var_redo - intermed_var[(len(rewriting_prompts))*(i) : (len(rewriting_prompts))*(i+1)] )))

                log_probs_redo = torch.log_softmax(ln_f(full_repr_redo) @ lm_w + lm_b, dim=2)

                cur_diff = torch.abs(log_probs_redo - log_probs[(len(rewriting_prompts))*(i) : (len(rewriting_prompts))*(i+1)])
                # print("DEBUG: log probs", torch.isclose(
                #     log_probs[(len(rewriting_prompts))*(i) : (len(rewriting_prompts))*(i+1)], 
                #     log_probs_redo).all(),
                #     torch.max(cur_diff))
                loss_redo = torch.gather(
                    log_probs_redo,
                    2,
                    torch.where(cur_rewriting_targets != -100, cur_rewriting_targets, 0).unsqueeze(2),
                ).squeeze(2)
                # print("loss_redo.shape", loss_redo.shape)
                # print("DEBUG: loss", torch.isclose(
                #     loss[(len(rewriting_prompts))*(i) : (len(rewriting_prompts))*(i+1)], 
                #     loss_redo).all(),
                #     torch.max(torch.abs(loss_redo - loss[(len(rewriting_prompts))*(i) : (len(rewriting_prompts))*(i+1)])))
                # mask_redo = (cur_rewriting_targets != -100).float() # TODO: check
                mask_redo = (cur_rewriting_targets != -100).float()
                assert (torch.isclose(
                    mask[(len(rewriting_prompts))*(i) : (len(rewriting_prompts))*(i+1)], 
                    mask_redo).all())
                # Aggregate total losses
                nll_loss_each_redo = -(loss_redo * mask_redo).sum(1) / target_ids.size(0)
                # print("i", i, "->", nll_loss_each_redo[:10])
                nll_loss_dict[i] = nll_loss_each_redo.mean()
                # print("DEBUG: nll_loss_each", torch.isclose(
                #     nll_loss_each[(len(rewriting_prompts))*(i) : (len(rewriting_prompts))*(i+1)], 
                #     nll_loss_each_redo).all(),
                #     torch.max(torch.abs(nll_loss_each_redo - nll_loss_each[(len(rewriting_prompts))*(i) : (len(rewriting_prompts))*(i+1)])))

            # make sure the losses are close together
            print(f"DEBUG [{request['target_new']['str']}] VS [{request['target_anti']['str']}]")
            print("DEBUG DIFFERENCE", nll_loss - (nll_loss_dict[0] - nll_loss_dict[1]))
            assert torch.isclose(nll_loss, (nll_loss_dict[0] - nll_loss_dict[1]), atol=1e-2), \
                f"Severe mismatch between {(nll_loss, (nll_loss_dict[0] - nll_loss_dict[1]))}"

        
        kl_loss = hparams.kl_factor * torch.nn.functional.kl_div(
            kl_distr_init, kl_log_probs, log_target=True, reduction="batchmean"
        )
        weight_decay = hparams.v_weight_decay * (
            torch.norm(delta) / torch.norm(target_init) ** 2
        )
        # weight_decay = hparams.v_weight_decay * torch.norm(delta) ** 2
        print('DEBUG ratio:', nll_loss, kl_loss, weight_decay)
        loss = nll_loss + kl_loss + weight_decay
        if not flip_loss:
            print(
                f"loss {np.round(loss.item(), 3)} = {np.round(nll_loss.item(), 3)} + {np.round(kl_loss.item(), 3)} + {np.round(weight_decay.item(), 3)} "
                f"avg prob of [{request['target_new']['str']}] VS [{request['target_anti']['str']}]"
                f"{torch.exp(-nll_loss_each).mean().item()}"
            )
        else:
            print(
                f"loss {np.round(loss.item(), 3)} = {np.round(nll_loss.item(), 3)} + {np.round(kl_loss.item(), 3)} + {np.round(weight_decay.item(), 3)} "
                f"avg prob of [{request['target_new']['str']}] VS [{request['target_anti']['str']}]"
                f"{torch.exp(nll_loss_each).mean().item()}"
            )
        
        if not flip_loss and loss < 5e-2:
            break

        if flip_loss and loss < -50: # TODO: validate this stopping condition
            break

        if it == hparams.v_num_grad_steps - 1:
            break

        # Backpropagate
        loss.backward()
        opt.step()

        # Project within L2 ball
        max_norm = hparams.clamp_norm_factor * target_init.norm()
        if delta.norm() > max_norm:
            with torch.no_grad():
                delta[...] = delta * max_norm / delta.norm()

    target = target_init + delta
    print(
        f"Init norm {target_init.norm()} | Delta norm {delta.norm()} | Target norm {target.norm()}"
    )

    return target


def get_module_input_output_at_words(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer: int,
    context_templates: List[str],
    words: List[str],
    module_template: str,
    fact_token_strategy: str,
) -> Tuple[torch.Tensor]:
    """
    Retrieves detached representations for a word at the input and
    output of a particular layer module.
    """

    word_repr_args = dict(
        model=model,
        tok=tok,
        layer=layer,
        module_template=module_template,
    )
    if "subject_" in fact_token_strategy and fact_token_strategy.index("subject_") == 0:
        context_info = dict(
            context_templates=context_templates,
            words=words,
        )
        subtoken = fact_token_strategy[len("subject_") :]
        l_input, l_output = repr_tools.get_reprs_at_word_tokens(
            track="both", subtoken=subtoken, **context_info, **word_repr_args
        )
    elif fact_token_strategy == "last":
        raise Exception("This is definitely bugged, fix it.")
        context_info = dict(
            contexts=[
                tmp[i].format(words[i]) for i, tmp in enumerate(context_templates)
            ],
            idxs=[000000],
        )
        l_input, l_output = repr_tools.get_reprs_at_idxs(
            track="both", **context_info, **word_repr_args
        )
    else:
        raise ValueError(f"fact_token={fact_token_strategy} not recognized")

    return l_input.detach(), l_output.detach()


def find_fact_lookup_idx(
    prompt: str,
    subject: str,
    tok: AutoTokenizer,
    fact_token_strategy: str,
    verbose=True,
) -> int:
    """
    Computes hypothesized fact lookup index given a sentence and subject.
    """

    ret = None
    if fact_token_strategy == "last":
        ret = -1
    elif (
        "subject_" in fact_token_strategy and fact_token_strategy.index("subject_") == 0
    ):
        ret = repr_tools.get_words_idxs_in_templates(
            tok=tok,
            context_templates=[prompt],
            words=[subject],
            subtoken=fact_token_strategy[len("subject_") :],
        )[0][0]
    else:
        raise ValueError(f"fact_token={fact_token_strategy} not recognized")

    sentence = prompt.format(subject)
    if verbose:
        print(
            f"Lookup index found: {ret} | Sentence: {sentence} | Token:",
            tok.decode(tok(sentence)["input_ids"][ret]),
        )

    return ret
