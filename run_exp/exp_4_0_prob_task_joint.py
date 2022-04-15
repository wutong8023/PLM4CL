"""


Author: Tong
Time: --2021
"""


def generate_cmd():
    cmd_list = []
    info_list = []
    
    for model in ["joint_nlp"]:
        for dataset in ["seq-clinc150"]:
            for seed in range(1):
                seed = seed * 100
                for ptm in ["bert"]:
                    for filter_rate in [1]:
                        for lr in [0.00001]:
                            for epoch in [30]:
                                for p_type in ["proto"]:
                                    for fix_layers in ["all", "bottom_n", "none"]:
                                        for fix_layers_n in [8]:
                                            for feature_layers in ["last", "top_n"]:
                                                for feature_layers_n in [4]:
                                                    info = "{description}_{var}_{var1}_{var2}".format(description="e40",
                                                                                                      var=p_type,
                                                                                                      var1=ptm,
                                                                                                      var2="joint")
                                                    cmd = 'python3 -m utils.main --info {info} --seed {seed} ' \
                                                          '--model {model} --area NLP --dataset {dataset} ' \
                                                          '--csv_log --tensorboard --lr {lr} --ptm {ptm} ' \
                                                          '--prob_all_tasks --increment_joint ' \
                                                          '--eval_freq 1 --prob_type {p_type} --filter_rate {filter_rate} ' \
                                                          '--batch_size 32 --n_epochs {epoch} ' \
                                                          '--fix_layers {fix_l} --feature_layers {feat_l} ' \
                                                          '--fix_layers_n {fix_l_n} --feature_layers_n {feat_l_n} '.format(
                                                        model=model,
                                                        dataset=dataset,
                                                        seed=seed,
                                                        ptm=ptm,
                                                        info=info,
                                                        filter_rate=filter_rate,
                                                        lr=lr,
                                                        p_type=p_type,
                                                        epoch=epoch,
                                                        fix_l=fix_layers,
                                                        feat_l=feature_layers,
                                                        fix_l_n=fix_layers_n,
                                                        feat_l_n=feature_layers_n,
                                                        )
                                                    info_list.append(info)
                                                    cmd_list.append(cmd)
    
    return cmd_list, info_list
