# run memit for various models and datasets
# MODEL_NAME = "stanfordnlp/backpack-gpt2"
# 'gpt2-xl' 
# "EleutherAI/gpt-j-6B"
# "EleutherAI/pythia-160m"
save_dir = None 
# log_dir = 'log_memit_results_layer_tuning'
log_dir = 'log_memit_results_val_10'

ALG_NAME = "MEMIT"
import json 
import yaml 
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import argparse
from util import nethook
from experiments.py.demo import load_alg, HPARAMS_DIR, print_loud
import os
import sys
sys.path.append('/nlp/scr/sachen/backpack_project/backpack-guarantees')
import utils

########## eval code copied from elsewhere ##########
EVAL_BATCH_SIZE = 1
class ScoreEvaluator:
  def __init__(self, args, model, tokenizer, eval_type='suffix', loss_type='good', threshold=None, normalize='token'):
    self.args = args
    self.model = model
    self.tokenizer = tokenizer
    self.threshold = threshold
    self.normalize = normalize
    self.data = [json.loads(x) for x in open(args['evaluation_set'])]
    self.loss_type = loss_type
    # print(">> INITIALIZING score evaluator")
    # print(">> normalize", normalize)
    # print(">> args['evaluation_set']", args['evaluation_set'])
    # print(">> eval_type", eval_type)
    if eval_type == 'suffix':
      self.batches = [x for x in utils.suffix_batch_iterator(self.data, tokenizer, device=model.device, batch_size=EVAL_BATCH_SIZE)]
    elif eval_type == 'unconditional':
      self.batches = [x for x in utils.unconditional_batch_iterator(self.data, tokenizer, device=model.device, batch_size=EVAL_BATCH_SIZE)]

  def evaluate(self):
    total_score = 0
    total_elts = 0
    for batch in tqdm(self.batches, desc='scoring'):
      output = self.model(batch['input_ids']).logits
      target = utils.target_of_indices(batch['input_ids'])
      scores = utils.score_suffix(output, target, batch['loss_mask'], reduction='none', reduce=False, loss_type=self.loss_type)
      if self.normalize == 'token':
        total_elts += torch.sum(batch['loss_mask']).item()
      elif self.normalize == 'example':
        total_elts += torch.sum((torch.sum(batch['loss_mask'], dim=-1)>0)).item()
      if self.threshold is not None:
        if self.normalize == 'example':
          scores = torch.sum(scores, dim=-1)
        scores = scores > self.threshold # failure rate
      total_score += torch.sum(scores).item()
    return total_score/total_elts

class PairEvaluator:

  def __init__(self, args, model, tokenizer, eval_type='suffix', diff_type='max_ratio', loss_type='balance', threshold=None, normalize='token'):
    self.args = args
    self.model = model
    self.normalize = normalize
    self.tokenizer = tokenizer
    self.data = [json.loads(x) for x in open(args['evaluation_set'])]
    self.threshold = threshold
    self.loss_type = loss_type
    if eval_type == 'suffix':
      self.batches = [x for x in utils.pair_suffix_batch_iterator(self.data, tokenizer, device=model.device, batch_size=EVAL_BATCH_SIZE)]

  def evaluate(self):
    total_score = 0
    total_elts = 0
    for batch in self.batches:
      output1 = self.model(batch['input_ids1']).logits
      target1 = utils.target_of_indices(batch['input_ids1'])
      output2 = self.model(batch['input_ids2']).logits
      target2 = utils.target_of_indices(batch['input_ids2'])
      #return utils.score_pair_suffix(output1, target1, output2, target2, batch['loss_mask1'], batch['loss_mask2'], self.loss_type).item()
      scores = utils.score_pair_suffix(output1, target1, output2, target2, batch['loss_mask1'], batch['loss_mask2'], self.loss_type, reduce=False)
      #if self.normalize == 'token':
      #  total_elts += torch.sum(batch['loss_mask']).item()
      if self.normalize == 'example':
        total_elts += torch.sum((torch.sum(batch['loss_mask1'], dim=-1)>0)).item() # same for both 1 and 2
      #if self.threshold is not None:
      #  scores = scores > self.threshold # failure rate
      if self.threshold is not None:
        scores = scores > self.threshold # failure rate
        #if self.normalize == 'example':
        #  scores = torch.sum(scores, dim=-1)
      scores = torch.sum(scores).item()
      total_score += scores
    return total_score/total_elts

def get_intervention_eval_class(config):
  if config['training']['suffix_pair']:
    return PairEvaluator
  else:
    return ScoreEvaluator

def eval_model_on_config(model_to_eval, config, cached_general_score={}):
  loss_type = config['training']['loss_type']

  # Build the validation function
  degredation_targeted_path = config['validation']['degredation_targeted_path']
  degredation_general_path = config['validation']['degredation_general_path']
  intervention_eval_path = config['validation']['intervention_eval_path']

  degredation_targeted_path = '/nlp/scr/sachen/backpack_project/backpack-guarantees/' + degredation_targeted_path
  degredation_general_path = '/nlp/scr/sachen/backpack_project/backpack-guarantees/' + degredation_general_path
  intervention_eval_path = '/nlp/scr/sachen/backpack_project/backpack-guarantees/' + intervention_eval_path

  threshold = config['threshold']
  normalize = config['validation']['eval_normalization']

  intervention_eval_class = get_intervention_eval_class(config)
  intervention_evaluator = intervention_eval_class(
      {'evaluation_set':intervention_eval_path}, model_to_eval, tok, loss_type=loss_type, threshold=threshold, normalize=normalize)
  rest_evaluator = ScoreEvaluator(
      {'evaluation_set':degredation_targeted_path},
      model_to_eval, tok, eval_type='unconditional', normalize='token')
  general_evaluator = ScoreEvaluator(
      {'evaluation_set':degredation_general_path},
      model_to_eval, tok, eval_type='unconditional', normalize='token')
  
  model_to_eval.eval()

  intervention_score = intervention_evaluator.evaluate()
  rest_of_prompt_score = rest_evaluator.evaluate()

  if len(cached_general_score) > 0:
    assert degredation_general_path == cached_general_score['degredation_general_path']
    general_score = cached_general_score['general_score']
  else:
    general_score = general_evaluator.evaluate()

    # cache results
    print("WARNING: USING CACHED RESULTS FOR GENERAL SCORE")
    cached_general_score['degredation_general_path'] = degredation_general_path
    cached_general_score['general_score'] = general_score

  return {
    'intervention_score': intervention_score,
    'general_score': general_score,
    'rest_of_prompt_score': rest_of_prompt_score,
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
        use_balance=use_balance
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

  argp.add_argument('-d', '--dataset_names')
  argp.add_argument('-s', '--subject_types')

  args = argp.parse_args()

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


  # load configs
  config_list = []
  for dataset_name in dataset_names:
    cfg_dataset_name = 'verb' if dataset_name == 'verbs' else dataset_name

    for subject_type in subject_types:
      config_list.append(
        {
          'dataset_path': f'/nlp/scr/sachen/backpack_project/backpack-guarantees/memit_data/{dataset_name}-{subject_type}.jsonl',
          'eval_ex_yaml': f'/nlp/scr/sachen/backpack_project/backpack-guarantees/configs/mini_merge/{cfg_dataset_name}_full_0.01.yaml',
          'flip_loss': (dataset_name == 'stereoset'),
          'use_balance': (dataset_name == 'gender')
        }
      )
  print("config_list", config_list)
  print('='*20)
  print('='*20)
  print('='*20)


  all_results = []

  if MODEL_NAME == 'stanfordnlp/backpack-gpt2':
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
    model, tok = (
      AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=(torch.float16 if any([x in MODEL_NAME for x in ["20b", "2.8b", '6.9b']]) else None),
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
    dataset_path = exp_config_dict['dataset_path']
    config_path_for_eval = exp_config_dict['eval_ex_yaml']
    config_dict = yaml.safe_load(open(config_path_for_eval))

    exp_name = MODEL_NAME.split('/')[-1] + '__' + dataset_path.split('/')[-1].split(".jsonl")[0]
    if args.noedit:
      eval_results = eval_model_on_config(model, config_dict, eval_general_cache)
      with open(f"{log_dir}/noedit.{exp_name}.json", 'w') as fh:
        print(json.dumps(eval_results), file=fh)
      continue 

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
      print("Results already exist, skipping", exp_name)
      continue 



    # Restore fresh copy of model
    try:
        with torch.no_grad():
            for k, v in orig_weights.items():
                nethook.get_parameter(model, k)[...] = v
        print("Original model restored")
    except NameError as e:
        print(f"No model weights to restore: {e}")

    # Execute rewrite
    request = [json.loads(line) for line in open(dataset_path)]
    model_new, orig_weights = modified_demo_model_editing(
        model, tok, request, alg_name=ALG_NAME, 
        flip_loss=exp_config_dict['flip_loss'],
        use_balance=exp_config_dict['use_balance'],
        override_params=override_params
    )

    eval_results = eval_model_on_config(model, config_dict)
    eval_results['override_params'] = override_params

    all_results.append(eval_results)

    if save_dir is not None:
      model.save_pretrained(f"{save_dir}/{exp_name}")

    with open(f"{log_dir}/{exp_name}.json", 'w') as fh:
      print(json.dumps(eval_results), file=fh)
  print()
  print(config_list)
  print(all_results)
