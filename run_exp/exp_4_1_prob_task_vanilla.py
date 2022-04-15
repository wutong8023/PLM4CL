"""


Author: Tong
Time: --2021
"""


def generate_cmd():
    cmd_list = []
    info_list = []
    
    for model in ["vanilla_nlp"]:
        # for dataset in ["seq-clinc150", "seq-maven", "seq-webred"]:
        for dataset in ["seq-clinc150"]:
            # for dataset in ["seq-clinc150", "online-maven", "online-webred"]:
            for seed in range(1):
                seed = seed * 100
                for ptm in ["bert", "xlnet", "gpt2", "albert", "roberta"]:
                    for filter_rate in [1]:
                        for lr in [0.00001]:
                            for epoch in [20]:
                                for p_type in ["proto"]:
                                    for fix_layers in ["none"]:
                                        # for fix_layers_n in [8]:
                                        for feature_layers in ["specified_n"]:
                                            for feature_layers_n in [9, 7]:
                                                for auto_layer in ["--auto_layer"]:
                                                    # --fix_layers_n {fix_l_n}
                                                    info = "{description}_{var}_{var1}_{var2}".format(description="e41",
                                                                                                      var=p_type,
                                                                                                      var1=ptm,
                                                                                                      var2="vnla")
                                                    cmd = 'python3 -m utils.main --info {info} --seed {seed} ' \
                                                          '--model {model} --area NLP --dataset {dataset} ' \
                                                          '--csv_log --tensorboard --lr {lr} --ptm {ptm} ' \
                                                          '--prob_all_tasks ' \
                                                          '--eval_freq 1 --prob_type {p_type} --filter_rate {filter_rate} ' \
                                                          '--batch_size 32 --n_epochs {epoch} ' \
                                                          '--fix_layers {fix_l} --feature_layers {feat_l} ' \
                                                          '--feature_layers_n {feat_l_n} ' \
                                                          '{auto_l}'.format(model=model,
                                                                            dataset=dataset,
                                                                            seed=seed,
                                                                            ptm=ptm,
                                                                            info=info,
                                                                            filter_rate=filter_rate,
                                                                            lr=lr,
                                                                            fix_l=fix_layers,
                                                                            feat_l=feature_layers,
                                                                            # fix_l_n=fix_layers_n,
                                                                            feat_l_n=feature_layers_n,
                                                                            p_type=p_type,
                                                                            epoch=epoch,
                                                                            auto_l=auto_layer)
                                                    info_list.append(info)
                                                    cmd_list.append(cmd)
    
    return cmd_list, info_list
