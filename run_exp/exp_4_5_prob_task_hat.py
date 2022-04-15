"""
epoch selection

Author: Tong
Time: 25-04-2021
"""

map = {"seq-clinc150": "c", "seq-maven": "m", "seq-webred": "w"}


def generate_cmd():
    cmd_list = []
    info_list = []
    
    for model in ["hat_nlp"]:
        for ptm in ["xlnet"]:
            for dataset in ["seq-webred"]:
                for seed in range(1):
                    seed = seed * 100
                    for filter_rate in [1]:
                        for lamb in [0.75]:
                            for smax in [400]:
                                for clipgrad in [1000]:
                                    for thres_cosh in [50]:
                                        for thres_emb in [6]:
                                            for prob_l in [-1]:
                                                for lr in [0.00001]:
                                                    for p_type in ["proto"]:
                                                        for epoch in [20]:  # todo
                                                            info = "{description}_{var1}_{var2}".format(
                                                                description="e45",
                                                                var=p_type,
                                                                var1=ptm,
                                                                var2="hat")
                                                            cmd = 'python3 -m utils.main --info {info} ' \
                                                                  '--seed {seed} ' \
                                                                  '--model {model} --area NLP --dataset {dataset} ' \
                                                                  '--csv_log --tensorboard --lr {lr} --ptm {ptm} ' \
                                                                  '--eval_freq 1 --prob_l {prob_l} ' \
                                                                  '--filter_rate {filter_rate} ' \
                                                                  '--batch_size 32 ' \
                                                                  '--lamb {lamb} ' \
                                                                  '--smax {smax} ' \
                                                                  '--prob_type {p_type} --prob_all_tasks ' \
                                                                  '--clipgrad {cg} ' \
                                                                  '--thres_cosh {tc} ' \
                                                                  '--thres_emb {te} ' \
                                                                  '--n_epochs {epoch}'.format(model=model,
                                                                                              dataset=dataset,
                                                                                              seed=seed,
                                                                                              ptm=ptm,
                                                                                              info=info,
                                                                                              lamb=lamb,
                                                                                              smax=smax,
                                                                                              cg=clipgrad,
                                                                                              tc=thres_cosh,
                                                                                              te=thres_emb,
                                                                                              p_type=p_type,
                                                                                              filter_rate=filter_rate,
                                                                                              prob_l=prob_l,
                                                                                              lr=lr,
                                                                                              epoch=epoch)
                                                            info_list.append(info)
                                                            cmd_list.append(cmd)
    
    return cmd_list, info_list
