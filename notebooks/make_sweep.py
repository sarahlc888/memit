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
]
added_model_names = ['gpt-j']

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
    if '6.9b' in model_name:
        return 'jag-standard'
    else:
        return 'jag-lo'
    
# UPDATED VERSION
# def model_to_jags(model_name):
#     if '6.9b' in model_name or '2.8b' in model_name or '1b' in model_name or '160m' in model_name:
#         return ['jagupard37', 'jagupard38', 'jagupard39']
#     elif '1.4b' in model_name:
#         return ['jagupard32', 'jagupard33', 'jagupard34', 'jagupard35', 'jagupard36']
#     elif '410m' in model_name:
#         return ['jagupard30', 'jagupard31', ]
#     elif '70m' in model_name or 'backpack' in model_name:
#         return ['jagupard28', 'jagupard29', ]
#     else:
#         raise ValueError
# RECOVERY VERSION
def model_to_jags(model_name):
    if '6.9b' in model_name or '2.8b' in model_name:
        return ['jagupard37', 'jagupard38', 'jagupard39']
    elif '1.4b' in model_name or 'gpt-j' in model_name:
        return ['jagupard32', 'jagupard33', 'jagupard34', 'jagupard35', 'jagupard36']
    elif '410m' in model_name or '1b' in model_name:
        return ['jagupard30', 'jagupard31', ]
    elif '70m' in model_name or 'backpack' in model_name or '160m' in model_name:
        return ['jagupard28', 'jagupard29', ]
    else:
        raise ValueError


def model_name_to_short(model_name):
    if model_name == 'backpack-gpt2':
        return 'bpk' 
    if 'pythia' in model_name:
        return model_name.split('-')[1]
    return model_name

def make_noedit_run(
        sweep_script_dir,
        log_dir,
    ):
    model_names = [
        'backpack-gpt2',
        'pythia-70m',
        'pythia-160m',
        'pythia-410m',
        'pythia-1b',
        'pythia-1.4b',
        'pythia-2.8b',
        'pythia-6.9b'
    ]
    for model_name in model_names + added_model_names:

        cmd = (
            f'python3 run_memit.py "{model_name_to_full[model_name]}" '
            '--noedit '
            f'--log_dir {log_dir} '
            f">> {sweep_script_dir}/logs/log.noedit.{model_name}.txt"
        )
        

        cur_model_name = model_name_to_short(model_name)
        with open(f"{sweep_script_dir}/scripts/noedit_{cur_model_name}.sbatch", "w") as fh:
            print(
                get_sbatch_header(
                    run_name=f'{cur_model_name}-noedit', 
                    partition=model_to_queue(model_name), 
                    nodelist=np.random.choice(model_to_jags(model_name)),
                    log_output_dir=f"{sweep_script_dir}/logs"
                ),
                file=fh
            )
            run_cmd = (
                f"srun --unbuffered run_as_child_processes '{cmd}'"
            )
            print(run_cmd, file=fh)
        print(f"sbatch {sweep_script_dir}/scripts/noedit_{cur_model_name}.sbatch")
    return None

def make_sweep_valn_granular(
        sweep_script_dir,
        log_dir,
        n=10, 
    ):
    """n is configs_per_model_and_dataset to use"""
    v_num_grad_steps = 20

    for dataset_name in dataset_names:
        for model_name in model_names:
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
                with open(f"{sweep_script_dir}/scripts/{cur_model_name}_{dataset_name}_{i}.sbatch", "w") as fh:
                                        
                    print(
                        get_sbatch_header(
                            run_name=f'{cur_model_name}_{dataset_name[:3]}_{i}-sweep', 
                            partition=model_to_queue(model_name), 
                            nodelist=np.random.choice(model_to_jags(model_name)),
                            log_output_dir=f"{sweep_script_dir}/logs"
                        ),
                        file=fh
                    )
                    run_cmd = (
                        f"srun --unbuffered run_as_child_processes '{cmd}' "
                        f">> {sweep_script_dir}/logs/log.{model_name}.{dataset_name}.{i}.txt"
                    )
                    print(run_cmd, file=fh)

    for dataset_name in dataset_names:
        for model_name in added_model_names:
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
                with open(f"{sweep_script_dir}/scripts/{cur_model_name}_{dataset_name}_{i}.sbatch", "w") as fh:
                                        
                    print(
                        get_sbatch_header(
                            run_name=f'{cur_model_name}_{dataset_name[:3]}_{i}-sweep', 
                            partition=model_to_queue(model_name), 
                            nodelist=np.random.choice(model_to_jags(model_name)),
                            log_output_dir=f"{sweep_script_dir}/logs"
                        ),
                        file=fh
                    )
                    run_cmd = (
                        f"srun --unbuffered run_as_child_processes '{cmd}' "
                        f">> {sweep_script_dir}/logs/log.{model_name}.{dataset_name}.{i}.txt"
                    )
                    print(run_cmd, file=fh)

def make_sweep_valn_granular_all(
        sweep_script_dir,
        log_dir,
        n=10, 
    ):
    """n is configs_per_model_and_dataset to use"""
    v_num_grad_steps = 20

    for dataset_name in dataset_names:
        for model_name in model_names:
            mom2_update_weights = []
            clamp_norm_factors = []
            kl_factors = []
            for i in range(n):
                mom2_update_weight = np.random.randint(9000, 75000)
                clamp_norm_factor = 10 ** np.random.uniform(-1.35, 0)
                kl_factor = np.random.uniform(low=0.001, high=0.1)
                mom2_update_weights.append(mom2_update_weight)
                clamp_norm_factors.append(clamp_norm_factor)
                kl_factors.append(kl_factor)

                cmd = (
                    f'python3 run_memit.py "{model_name_to_full[model_name]}" --v_num_grad_steps {v_num_grad_steps} '
                    f'--clamp_norm_factor {clamp_norm_factor} --mom2_update_weight {mom2_update_weight} '
                    f'--kl_factor {kl_factor} '
                    f'--dataset_names {dataset_name} --subject_types true_subject,prefix_subject '
                    f'--log_dir {log_dir}'
                )
                

                cur_model_name = model_name_to_short(model_name)
                with open(f"{sweep_script_dir}/scripts/{cur_model_name}_{dataset_name}_{i}.sbatch", "w") as fh:
                    print(
                        get_sbatch_header(
                            run_name=f'{cur_model_name}_{dataset_name[:3]}_{i}-sweep', 
                            partition=model_to_queue(model_name), 
                            nodelist=np.random.choice(model_to_jags(model_name)),
                            log_output_dir=f"{sweep_script_dir}/logs"
                        ),
                        file=fh
                    )
                    run_cmd = (
                        f"srun --unbuffered run_as_child_processes '{cmd}' "
                        f">> {sweep_script_dir}/logs/log.{model_name}.{dataset_name}.{i}.txt"
                    )
                    print(run_cmd, file=fh)
            with open(f"{sweep_script_dir}/scripts/{cur_model_name}_{dataset_name}_ALL.sbatch", "w") as fh:
                print(
                    get_sbatch_header(
                        run_name=f'{cur_model_name}_{dataset_name[:3]}_ALL-sweep', 
                        partition=model_to_queue(model_name), 
                        nodelist=np.random.choice(model_to_jags(model_name)),
                        log_output_dir=f"{sweep_script_dir}/logs"
                    ),
                    file=fh
                )
                for i in range(n):
                    mom2_update_weight = mom2_update_weights[i]
                    clamp_norm_factor = clamp_norm_factors[i]
                    kl_factor = kl_factors[i]

                    cmd = (
                        f'python3 run_memit.py "{model_name_to_full[model_name]}" --v_num_grad_steps {v_num_grad_steps} '
                        f'--clamp_norm_factor {clamp_norm_factor} --mom2_update_weight {mom2_update_weight} '
                        f'--kl_factor {kl_factor} '
                        f'--dataset_names {dataset_name} --subject_types true_subject,prefix_subject '
                        f'--log_dir {log_dir}'
                    )

                    run_cmd = (
                        f"srun --unbuffered run_as_child_processes '{cmd}' "
                        f">> {sweep_script_dir}/logs/log.{model_name}.{dataset_name}.{i}.txt"
                    )
                    print(run_cmd, file=fh)
    return None

def make_sweep_val_10(
        configs_per_model_and_dataset=10, 
        sweep_script_dir='sbatches_val_10_sweep',
        log_dir='log_memit_results_val_10'):
    v_num_grad_steps = 20

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
        'pythia-6.9b'
    ]
    model_name_to_full = {
        'backpack-gpt2': "stanfordnlp/backpack-gpt2",
        'pythia-70m': "EleutherAI/pythia-70m",
        'pythia-160m': "EleutherAI/pythia-160m",
        'pythia-410m': "EleutherAI/pythia-410m",
        'pythia-1b': "EleutherAI/pythia-1b",
        'pythia-1.4b': "EleutherAI/pythia-1.4b",
        'pythia-2.8b': "EleutherAI/pythia-2.8b",
        'pythia-6.9b': "EleutherAI/pythia-6.9b"
    }

    for dataset_name in dataset_names:
        for model_name in model_names:
            cmds = []

            for _ in range(configs_per_model_and_dataset):
                mom2_update_weight = np.random.randint(9000, 75000)
                clamp_norm_factor = np.random.uniform(low=0.1, high=0.8)
                # np.random.triangular(0.1, 0.3, 0.85, 50) 
                kl_factor = np.random.uniform(low=0.001, high=0.1)
                cmd = (
                    f'python3 run_memit.py "{model_name_to_full[model_name]}" --v_num_grad_steps {v_num_grad_steps} '
                    f'--clamp_norm_factor {clamp_norm_factor} --mom2_update_weight {mom2_update_weight} '
                    f'--kl_factor {kl_factor} '
                    f'--dataset_names {dataset_name} --subject_types true_subject,prefix_subject '
                    f'--log_dir {log_dir}'
                )
                cmds.append(cmd)
            
            with open(f"{sweep_script_dir}/{model_name}_{dataset_name}.sh", "w") as fh:
                print('#!/bin/bash', file=fh)
                print(*cmds, sep='\n', file=fh)

    for model_name in model_names:
        with open(f"{sweep_script_dir}/runsweep-{model_name}.sbatch", "w") as fh:
            cur_model_name = model_name if model_name == 'backpack-gpt2' else model_name.split('-')[1]
            print(
                get_sbatch_header(
                    run_name=f'{cur_model_name}-sweep', partition='jag-standard', nodelist='jagupard31',
                    log_output_dir=sweep_script_dir
                ),
                file=fh
            )
            for dataset_name in dataset_names:
                run_cmd = (
                    f"srun --unbuffered run_as_child_processes 'bash {sweep_script_dir}/{model_name}_{dataset_name}.sh' "
                    f">> {sweep_script_dir}/log.{model_name}.{dataset_name}.txt"
                )
                print(run_cmd, file=fh)

    return cmds

if __name__ == '__main__':
    make_sweep_valn_granular(
        sweep_script_dir='sbatches_100523',
        log_dir='log_memit_100223',
        n=10, 
    )
    make_noedit_run(
        sweep_script_dir='sbatches_100523',
        log_dir='log_memit_100223',
    )

    # make_sweep_valn_granular(
    #     sweep_script_dir='sbatches_val_10_final_sweep',
    #     log_dir='log_memit_results_val_10_final',
    #     n=16, 
    # )
    # make_noedit_run(
    #     sweep_script_dir='sbatches_val_10_final_sweep',
    #     log_dir='log_memit_results_val_10_final',
    # )

    # make_sweep_valn_granular(
    #     sweep_script_dir='sbatches_val_12_sweep',
    #     log_dir='log_memit_results_val_12',
    #     n=16, 
    # )
    # make_noedit_run(
    #     sweep_script_dir='sbatches_val_12_sweep',
    #     log_dir='log_memit_results_val_12',
    # )

    # cmds = make_sweep_val_10(
    #     configs_per_model_and_dataset=20,
    #     sweep_script_dir='sbatches_val_20_sweep',
    #     log_dir='log_memit_results_val_20'
    #     )
    # cmds = make_sweep_val_10()

def make_sweep_8():
    v_num_grad_steps_vals = [20] # [np.random.randint(3, 50) for _ in range(2)]# [15, 25]
    clamp_norm_factor_vals = [np.random.uniform(low=0.1, high=0.8) for _ in range(4)]
    mom2_update_weight_vals = [np.random.randint(9000, 75000) for _ in range(2)]

    # add special ranges
    # mom2_update_weight_vals.append(np.random.randint(75000, 85000))
    # clamp_norm_factor_vals.extend([np.random.uniform(low=0.05, high=0.1) for _ in range(1)])
    #[int(np.random.uniform(0.5, 9) * 10**4) for _ in range(2)]
    # [round(10 ** np.random.uniform(4, 4.95)) for _ in range(2)] # [15000, 20000]

    mom2_update_weight_vals = sorted(mom2_update_weight_vals)
    clamp_norm_factor_vals = sorted(clamp_norm_factor_vals)

    print("mom2_update_weight_vals", mom2_update_weight_vals)
    print("clamp_norm_factor_vals", clamp_norm_factor_vals)

    cmds = []
    for mom2_update_weight in mom2_update_weight_vals:
        for clamp_norm_factor in clamp_norm_factor_vals:
            for v_num_grad_steps in v_num_grad_steps_vals:
                cmd = f'python3 run_memit.py "{{model_name}}" --v_num_grad_steps {v_num_grad_steps} --clamp_norm_factor {clamp_norm_factor} --mom2_update_weight {mom2_update_weight}'
                cmds.append(cmd)

    for model_name in [
        'EleutherAI/pythia-70m', 
        # 'EleutherAI/pythia-160m', 
        # 'EleutherAI/pythia-410m', 
        # 'EleutherAI/pythia-1b', 
        # 'EleutherAI/pythia-1.4b', 
        # 'EleutherAI/pythia-2.8b', 
        # 'EleutherAI/pythia-6.9b', 
        # "stanfordnlp/backpack-gpt2",
        ]: 
        for x in cmds:
            print(x.format(model_name=model_name))


