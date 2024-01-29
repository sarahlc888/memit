# run memit for various models and datasets

save_dir = None 

ALG_NAME = "MEMIT"
import json 
import yaml 
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from torch.cuda.amp import autocast
import argparse
from util import nethook
from experiments.py.demo import load_alg, HPARAMS_DIR, print_loud
import os
import sys
sys.path.insert(0, '/nlp/scr/sachen/backpack_project/backpack-guarantees')
import evaluate 


############ Eval code copied from ft_experiment.py ############
def get_intervention_eval_class(config):
  if config['training']['suffix_pair']:
    return evaluate.PairEvaluator
  else:
    return evaluate.ScoreEvaluator

def eval_model_on_config(model_to_eval, config, cached_general_score={}, test_mode=False):
  loss_type = config['training']['loss_type']

  # Build the validation function
  degredation_targeted_path = config['validation']['degredation_targeted_path']
  degredation_general_path = config['validation']['degredation_general_path']
  intervention_eval_path = config['validation']['intervention_eval_path']
  if 'hard_negative' in config['validation']:
    hard_negative_path = config['validation']['hard_negative']['hard_negative_path']
    hard_negative_eval_type = config['validation']['hard_negative']['eval_type']
    hard_negative_eval_normalize = "token" if hard_negative_eval_type == "unconditional" else "example"
  else:
    if test_mode:
      raise Exception("No hard negative eval in test mode")
    print("Warning: skipping hard negative eval")
    
  degredation_targeted_path = '/nlp/scr/sachen/backpack_project/backpack-guarantees/' + degredation_targeted_path
  degredation_general_path = '/nlp/scr/sachen/backpack_project/backpack-guarantees/' + degredation_general_path
  intervention_eval_path = '/nlp/scr/sachen/backpack_project/backpack-guarantees/' + intervention_eval_path

  threshold = config['threshold']
  normalize = config['validation']['eval_normalization']

  intervention_eval_class = get_intervention_eval_class(config)
  intervention_evaluator = intervention_eval_class(
      {'evaluation_set':intervention_eval_path}, model_to_eval, tok, loss_type=loss_type, threshold=threshold, normalize=normalize)
  if 'hard_negative' in config['validation']:
    hard_negative_path = '/nlp/scr/sachen/backpack_project/backpack-guarantees/' + hard_negative_path
    hard_negative_evaluator = evaluate.ScoreEvaluator(
      {'evaluation_set':hard_negative_path}, 
      model, tok, eval_type=hard_negative_eval_type, normalize=hard_negative_eval_normalize)
  rest_evaluator = evaluate.ScoreEvaluator(
      {'evaluation_set':degredation_targeted_path},
      model_to_eval, tok, eval_type='unconditional', normalize='token')
  general_evaluator = evaluate.ScoreEvaluator(
      {'evaluation_set':degredation_general_path},
      model_to_eval, tok, eval_type='unconditional', normalize='token')
  
  model_to_eval.eval()

  intervention_score = intervention_evaluator.evaluate()
  rest_of_prompt_score = rest_evaluator.evaluate()
  hard_negative_score = None
  if 'hard_negative' in config['validation']:
    hard_negative_score = hard_negative_evaluator.evaluate()

  if len(cached_general_score) > 0:
    print("WARNING: USING CACHED RESULTS FOR GENERAL SCORE") # note: this was originally in the wrong place
    assert degredation_general_path == cached_general_score['degredation_general_path']
    general_score = cached_general_score['general_score']
  else:
    general_score = general_evaluator.evaluate()

    # cache results
    cached_general_score['degredation_general_path'] = degredation_general_path
    cached_general_score['general_score'] = general_score

  return {
    'intervention_score': intervention_score,
    'general_score': general_score,
    'rest_of_prompt_score': rest_of_prompt_score,
    'hard_negative_score': hard_negative_score,
  }
#####################################################
from typing import Dict, List, Tuple
def modified_demo_model_editing(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    alg_name: str = "ROME",
    flip_loss: bool = False,
    use_balance: bool = False,
    use_anti: bool = False,
    override_params = {},
) -> Tuple[AutoModelForCausalLM, Dict[str, torch.Tensor]]:
    """
    Applies the selected model editing algorithm. Generates text both before and after
    for comparison of model behavior. Returns the updated model and the original values of
    weights that were changed.
    """

    nethook.set_requires_grad(True, model)

    RewritingParamsClass, apply_method, hparams_prefix, hparams_suffix = load_alg(
        alg_name
    )
    params_name = (
        HPARAMS_DIR
        / hparams_prefix
        / f"{model.config._name_or_path.replace('/', '_')}{hparams_suffix}.json"
    )

    print_loud(f"Retrieving {alg_name} hyperparameters")
    print("Loading from", params_name)
    hparams = RewritingParamsClass.from_json(params_name)
    for k in override_params:
      if override_params[k] is not None:
        setattr(hparams, k, override_params[k])

    print_loud(f"Applying {alg_name} to model")
    model_new, orig_weights = apply_method(
        model,
        tok,
        requests,
        hparams,
        return_orig_weights=True,
        flip_loss=flip_loss,
        use_balance=use_balance,
        use_anti=use_anti,
    )


    return model_new, orig_weights


if __name__ == '__main__':

  argp = argparse.ArgumentParser()
  argp.add_argument('model_name')
  argp.add_argument('-l', '--override_layers')
  argp.add_argument('--log_dir')
  argp.add_argument('--v_num_grad_steps', type=int)
  argp.add_argument('--clamp_norm_factor', type=float)
  argp.add_argument('--mom2_update_weight', type=int)
  argp.add_argument('--kl_factor', type=float)

  argp.add_argument('--noedit', dest='noedit', default=False, action='store_true')
  argp.add_argument('--test_mode', dest='test_mode', default=False, action='store_true')

  argp.add_argument('-d', '--dataset_names')
  argp.add_argument('-s', '--subject_types')
  argp.add_argument('--override_exp_name')
  argp.add_argument('--seed')

  args = argp.parse_args()

  if args.seed is not None:
    import random
    random.seed(args.seed)
    torch.manual_seed(args.seed)

  MODEL_NAME = args.model_name
  log_dir = 'log_memit_results_val_10' if args.log_dir is None else args.log_dir

  override_layers = args.override_layers
  if override_layers is not None:
    override_layers = [int(x) for x in override_layers.split(',')]
  override_params = {
    'layers': override_layers,
    'v_num_grad_steps': args.v_num_grad_steps,
    'clamp_norm_factor': args.clamp_norm_factor,
    'mom2_update_weight': args.mom2_update_weight,
    'kl_factor': args.kl_factor,
  }
  print("MODEL_NAME", MODEL_NAME)
  print("log_dir", log_dir)
  print("override_params", override_params)

  if args.dataset_names is not None:
    dataset_names = args.dataset_names.split(',')
  else:
    dataset_names = ['company', 'country', 'verbs', 'temporal', 'stereoset', 'gender']
  if args.subject_types is not None:
    subject_types = args.subject_types.split(',')
  else:
    subject_types = ['true_subject', 'prefix_subject']

  dname_cfg_map = {
    'company': 'company_ceo', 'country': 'country_capital', 'verbs': 'verb_conjugation', 
    'temporal': 'temporal', 'stereoset': 'stereoset', 'gender': 'pronoun_gender_bias'
  }

  # load configs
  config_list = []
  for dataset_name in dataset_names:
    cfg_dataset_name = dname_cfg_map[dataset_name]

    for subject_type in subject_types:

      if args.test_mode:
        f_dataset_name = 'verb' if dataset_name == 'verbs' else dataset_name
        eval_ex_yaml = (f'/nlp/scr/sachen/backpack_project/backpack-guarantees/configs/test/'
          f'test-stanfordnlp-backpack-gpt2-{f_dataset_name}-full-0.0001-0.yaml')
        dataset_path = f'/nlp/scr/sachen/backpack_project/backpack-guarantees/memit_data/test/{dataset_name}-{subject_type}.jsonl'
      else:
        eval_ex_yaml = f'/nlp/scr/sachen/backpack_project/backpack-guarantees/configs/{cfg_dataset_name}/backpack_sweep/stanfordnlp-backpack-gpt2-full.0.sweep.yaml'
        dataset_path = f'/nlp/scr/sachen/backpack_project/backpack-guarantees/memit_data/val/{dataset_name}-{subject_type}.jsonl'

      config_list.append(
        {
          'dataset_path': dataset_path,
          'eval_ex_yaml': eval_ex_yaml,
          'flip_loss': (dataset_name == 'stereoset'),
          'use_balance': (dataset_name == 'gender'),
          'use_anti': (dataset_name == 'verbs'),
        }
      )
  print("config_list", config_list)
  print('='*20)
  print('='*20)
  print('='*20)


  all_results = []

  if MODEL_NAME == 'stanfordnlp/backpack-gpt2':
    dtype = torch.float32
    tok = AutoTokenizer.from_pretrained('gpt2')
    config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, config=config, trust_remote_code=True).to("cuda")
    model.old_forward = model.forward
    def new_forward(*args, **kwargs):
      if 'input_ids' in kwargs:
        return model.old_forward(input_ids=kwargs['input_ids'])
      return model.old_forward(input_ids=args[0])
    # model.forward = lambda *args, **kwargs: model.old_forward(input_ids=kwargs['input_ids'])
    model.forward = new_forward

  else:
    dtype = torch.float16
    print("DTYPE", dtype)
    # TODO: float16 vs float32?
    # note: bfloat16 -> val sweep did not fill many leagues
    model, tok = (
      AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
        cache_dir="/u/scr/nlp/johnhew/data/huggingface/hub/"
      ).to("cuda"),
      AutoTokenizer.from_pretrained(MODEL_NAME),
    )
    if not hasattr(model.config, "n_embd"):
      assert hasattr(model.config, "hidden_size")
      model.config.n_embd = model.config.hidden_size
    if not hasattr(model.config, "n_positions"):
      assert hasattr(model.config, "max_position_embeddings")
      model.config.n_positions = model.config.max_position_embeddings
  tok.pad_token = tok.eos_token

  eval_general_cache = {}
  for i, exp_config_dict in enumerate(config_list):

    # Restore fresh copy of model (reset weights)
    try:
      with torch.no_grad():
        for k, v in orig_weights.items():
            nethook.get_parameter(model, k)[...] = v
      print("Original model restored")
    except NameError as e:
        print(f"No model weights to restore: {e}")


    model.eval() 


    dataset_path = exp_config_dict['dataset_path']
    config_path_for_eval = exp_config_dict['eval_ex_yaml']
    config_dict = yaml.safe_load(open(config_path_for_eval))

    exp_name = MODEL_NAME.split('/')[-1] + '__' + dataset_path.split('/')[-1].split(".jsonl")[0]

    # evaluate unedited model
    if dtype == torch.bfloat16:
      with autocast(dtype=dtype):
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False):
          noedit_eval_results = eval_model_on_config(model, config_dict, eval_general_cache, test_mode=args.test_mode)
    else:
      noedit_eval_results = eval_model_on_config(model, config_dict, eval_general_cache, test_mode=args.test_mode)
    if args.noedit:
      with open(f"{log_dir}/noedit.{exp_name}.json", 'w') as fh:
        print(json.dumps(noedit_eval_results), file=fh)
      continue 

    if args.override_exp_name is not None:
      exp_name = args.override_exp_name
    else:
      param_keys = ['layers', 'v_num_grad_steps', 'clamp_norm_factor', 'mom2_update_weight', 'kl_factor']
      for k in param_keys:
        if override_params[k] is not None:
          if type(override_params[k]) == list:
            exp_name += '__' + '-'.join([str(x) for x in override_params[k]])
          else:
            exp_name += '__' + str(override_params[k])
        else:
          exp_name += '__na'
        
    print("exp_name", exp_name)
    if os.path.exists(f"{log_dir}/{exp_name}.json"):
      print("Results already exist, skipping", exp_name, f"at {log_dir}/{exp_name}.json")
      continue 



    # Execute rewrite
    request = [json.loads(line) for line in open(dataset_path)]

    if dtype == torch.bfloat16:
      with autocast(dtype=dtype): 
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False):
          model_new, orig_weights = modified_demo_model_editing(
              model, tok, request, alg_name=ALG_NAME, 
              flip_loss=exp_config_dict['flip_loss'],
              use_balance=exp_config_dict['use_balance'],
              use_anti=exp_config_dict['use_anti'],
              override_params=override_params
          )
          eval_results = eval_model_on_config(model, config_dict, test_mode=args.test_mode)
    else:
      model_new, orig_weights = modified_demo_model_editing(
          model, tok, request, alg_name=ALG_NAME, 
          flip_loss=exp_config_dict['flip_loss'],
          use_balance=exp_config_dict['use_balance'],
          use_anti=exp_config_dict['use_anti'],
          override_params=override_params
      )
      eval_results = eval_model_on_config(model, config_dict, test_mode=args.test_mode)

    eval_results['override_params'] = override_params
    all_results.append(eval_results)

    if save_dir is not None:
      model.save_pretrained(f"{save_dir}/{exp_name}")

    with open(f"{log_dir}/{exp_name}.json", 'w') as fh:
      print(json.dumps({
        'noedit': noedit_eval_results,
        'edit': eval_results
      }), file=fh)
  print()
  print(config_list)
  print(all_results)
