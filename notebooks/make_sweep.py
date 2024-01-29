# Make scripts to run MEMIT hyperparameter sweep
# (Include cluster-specific logic that must be modified for compatibility)

import random
import numpy as np

np.random.seed(903)
random.seed(903)

def get_sbatch_header(run_name, partition='jag-standard', nodelist='jagupard31', log_output_dir='.', num_hrs=5):
    return f"""#!/bin/bash

#SBATCH --account=nlp
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --job-name={run_name}
#SBATCH --mem=32G
#SBATCH --open-mode=append
#SBATCH --output={log_output_dir}/slurm_log_{run_name}.out
#SBATCH --partition={partition}
#SBATCH --time=0-{num_hrs}
#SBATCH --nodelist={nodelist}

export LD_LIBRARY_PATH=/nlp/scr/sachen/miniconda3/envs/backpacks-env/lib/

source /nlp/scr/sachen/ref/switch-cuda/switch-cuda.sh 11.7
"""


dataset_names = [
    'company', 
    'country', 
    'verbs', 
    'temporal', 
    'stereoset', 
    'gender'
]
model_names = [
    'backpack-gpt2',
    'pythia-70m',
    'pythia-160m',
    'pythia-410m',
    'pythia-1b',
    'pythia-1.4b',
    'pythia-2.8b',
    'pythia-6.9b',
    'gpt-j'
]
model_name_to_full = {
    'backpack-gpt2': "stanfordnlp/backpack-gpt2",
    'pythia-70m': "EleutherAI/pythia-70m",
    'pythia-160m': "EleutherAI/pythia-160m",
    'pythia-410m': "EleutherAI/pythia-410m",
    'pythia-1b': "EleutherAI/pythia-1b",
    'pythia-1.4b': "EleutherAI/pythia-1.4b",
    'pythia-2.8b': "EleutherAI/pythia-2.8b",
    'pythia-6.9b': "EleutherAI/pythia-6.9b",
    'gpt-j': "EleutherAI/gpt-j-6B",
}

def model_to_queue(model_name):
    if '6.9b' in model_name or '2.8b' in model_name or '1.4b' in model_name or '1b' in model_name or 'gpt-j' in model_name:
        return 'jag-lo'
    else:
        return 'jag-standard'
    
def model_to_jags(model_name):
    # Map model to machine to use
    if '6.9b' in model_name or '2.8b' in model_name or '1.4b' in model_name or '1b' in model_name \
        or 'gpt-j' in model_name:
        return ['jagupard37', 'jagupard38', 'jagupard39']
    elif '410m' in model_name or '160m' in model_name or '70m' in model_name or 'backpack' in model_name:
        return ['jagupard32', 'jagupard33', 'jagupard34', 'jagupard35', 'jagupard36']
    else:
        raise ValueError

def model_name_to_short(model_name):
    if model_name == 'backpack-gpt2':
        return 'bpk' 
    if 'pythia' in model_name:
        return model_name.split('-')[1]
    return model_name

def make_noedit_run(sweep_script_dir, log_dir, script_dir='scripts', added_flags=[]):
    machine_choosing_index = 0
    for model_name in model_names:

        cmd = (
            f'python3 run_memit.py "{model_name_to_full[model_name]}" '
            '--noedit '
            f'--log_dir {log_dir} ' + ' '.join(added_flags) + f" >> {sweep_script_dir}/logs/log.noedit.{model_name}.txt"
        )
        

        cur_model_name = model_name_to_short(model_name)
        jag_options = model_to_jags(model_name)
        nodelist = jag_options[machine_choosing_index % len(jag_options)]
        machine_choosing_index += 1 
        
        with open(f"{sweep_script_dir}/{script_dir}/noedit_{cur_model_name}.sbatch", "w") as fh:
            print(
                get_sbatch_header(
                    run_name=f'{cur_model_name}-noedit', 
                    partition='jag-standard', 
                    nodelist=nodelist,
                    log_output_dir=f"{sweep_script_dir}/logs"
                ),
                file=fh
            )
            run_cmd = (
                f"srun --unbuffered run_as_child_processes '{cmd}'"
            )
            print(run_cmd, file=fh)
        print(f"sbatch {sweep_script_dir}/{script_dir}/noedit_{cur_model_name}.sbatch")
    return None

def make_sweep_valn_granular(sweep_script_dir, log_dir, n=10):
    """n is configs_per_model_and_dataset to use"""
    v_num_grad_steps = 20
    machine_choosing_index = 0

    fnames = []

    for model_name in model_names:
        for dataset_name in dataset_names:
            for i in range(n):
                mom2_update_weight = np.random.randint(9000, 75000)
                clamp_norm_factor = 10 ** np.random.uniform(-1.35, 0)
                kl_factor = np.random.uniform(low=0.001, high=0.1)
                cmd = (
                    f'python3 run_memit.py "{model_name_to_full[model_name]}" --v_num_grad_steps {v_num_grad_steps} '
                    f'--clamp_norm_factor {clamp_norm_factor} --mom2_update_weight {mom2_update_weight} '
                    f'--kl_factor {kl_factor} '
                    f'--dataset_names {dataset_name} --subject_types true_subject,prefix_subject '
                    f'--log_dir {log_dir}'
                )
                
                cur_model_name = model_name_to_short(model_name)
                jag_options = model_to_jags(model_name)
                nodelist = jag_options[machine_choosing_index % len(jag_options)]
                machine_choosing_index += 1 
                
                with open(f"{sweep_script_dir}/scripts/{cur_model_name}_{dataset_name}_{i}.sbatch", "w") as fh:
                                        
                    print(
                        get_sbatch_header(
                            run_name=f'{cur_model_name}_{dataset_name[:3]}_{i}-sweep', 
                            partition=model_to_queue(model_name), 
                            nodelist=nodelist,
                            log_output_dir=f"{sweep_script_dir}/logs"
                        ),
                        file=fh
                    )
                    run_cmd = (
                        f"srun --unbuffered run_as_child_processes '{cmd}' "
                        f">> {sweep_script_dir}/logs/log.{model_name}.{dataset_name}.{i}.txt"
                    )
                    print(run_cmd, file=fh)
                    fnames.append(f"{sweep_script_dir}/scripts/{cur_model_name}_{dataset_name}_{i}.sbatch")
    return fnames

if __name__ == '__main__':
    fnames = make_sweep_valn_granular(
        sweep_script_dir='sbatches_101023',
        log_dir='log_memit_101023',
        n=10, 
    )
