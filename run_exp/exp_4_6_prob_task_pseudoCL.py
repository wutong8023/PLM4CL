"""


Author: Tong
Time: 28-04-2021
"""
map = {"seq-clinc150": "c", "seq-maven": "m", "seq-webred": "w"}


def generate_cmd():
    cmd_list = []
    info_list = []
    for dataset in ["seq-clinc150", "seq-webred", "seq-maven"]:
        for model in ["per411_nlp", "per412_nlp", "per413_nlp"]:
            for seed in range(1):
                seed = seed * 100
                for ptm in ["bert"]:
                    for beta in [1]:
                        for alpha in [0.5]:
                            for lmd in [0.5]:
                                for epsilon in [0.5]:
                                    for pseudo_size in [2]:
                                        for lr in [0.00001]:
                                            for m_size in [200]:
                                                for epoch in [20]:
                                                    for p_type in ["proto"]:
                                                        for freq_eval in [1]:
                                                            if dataset.startswith("online"):
                                                                freq_eval = 100
                                                            info = "{description}_{var1}_{var2}_{var3}".format(
                                                                description="e46",
                                                                var=dataset,
                                                                var1=model.split("_")[0],
                                                                var2="beta",
                                                                var3=m_size)
                                                        
                                                        cmd = 'python3 -m utils.main --info {info} --seed {seed} ' \
                                                              '--model {model} --area NLP --dataset {dataset} ' \
                                                              '--buffer_size {m_size} ' \
                                                              '--csv_log --tensorboard --lr {lr} --minibatch_size 16 ' \
                                                              '--ptm {ptm} --freq_eval {freq_eval} ' \
                                                              '--lmd {lmd} --pseudo_size {pseudo_size} ' \
                                                              '--epsilon {epsilon} ' \
                                                              '--alpha {alpha} --beta {beta} --batch_size 16 ' \
                                                              '--eval_freq 1 --prob_type {p_type} --prob_all_tasks ' \
                                                              '--n_epochs {epoch}'.format(model=model,
                                                                                          alpha=alpha,
                                                                                          beta=beta,
                                                                                          lmd=lmd,
                                                                                          epsilon=epsilon,
                                                                                          pseudo_size=pseudo_size,
                                                                                          freq_eval=freq_eval,
                                                                                          seed=seed,
                                                                                          dataset=dataset,
                                                                                          ptm=ptm,
                                                                                          lr=lr,
                                                                                          m_size=m_size,
                                                                                          p_type=p_type,
                                                                                          epoch=epoch,
                                                                                          info=info)
                                                        info_list.append(info)
                                                        cmd_list.append(cmd)
    
    return cmd_list, info_list
