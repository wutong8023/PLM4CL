"""


Author: Tong
Time: 28-04-2021
"""
map = {"seq-clinc150": "c", "seq-maven": "m", "seq-webred": "w"}


def generate_cmd():
    cmd_list = []
    info_list = []
    for model in ["derpp_nlp"]:
        for dataset in ["seq-webred"]:
            for seed in range(1):
                seed = seed * 100
                for ptm in ["albert"]:
                    for alpha in [0.5]:
                        for lr in [0.00001]:
                            for beta in [1.0]:
                                for m_size in [500]:
                                    for epoch in [20]:
                                        for p_type in ["proto"]:
                                            info = "{description}_{var1}_{var2}_{var3}".format(description="e44",
                                                                                               var=p_type,
                                                                                               var1=ptm,
                                                                                               var2="derpp",
                                                                                               var3=m_size)
                                            
                                            cmd = 'python3 -m utils.main --info {info} --seed {seed} ' \
                                                  '--model {model} --area NLP --dataset {dataset} ' \
                                                  '--buffer_size {m_size} ' \
                                                  '--csv_log --tensorboard --lr {lr} --minibatch_size 16 --ptm {ptm} ' \
                                                  '--alpha {alpha} --beta {beta} --batch_size 16 ' \
                                                  '--eval_freq 1 --prob_type {p_type} --prob_all_tasks ' \
                                                  '--n_epochs {epoch}'.format(model=model,
                                                                              alpha=alpha,
                                                                              beta=beta,
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
