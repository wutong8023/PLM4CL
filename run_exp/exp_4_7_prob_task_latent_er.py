"""


Author: Tong
Time: 28-04-2021
"""


def generate_cmd():
    cmd_list = []
    info_list = []
    
    for model in ["latent_er_nlp"]:
        # for dataset in ["seq-clinc150", "seq-maven", "seq-webred"]:
        # for dataset in ["seq-clinc150"]:
            for dataset in ["seq-clinc150"]:
                for seed in range(1):
                    seed = seed * 100
                    # "albert", "roberta", "gpt2", "xlnet"
                    for ptm in ["bert", "albert", "roberta", "gpt2", "xlnet"]:
                        for lr in [0.00001]:
                            for beta in [1.0]:
                                for epoch in [20]:
                                    for p_type in ["proto"]:
                                        for m_size in [200]:
                                            # for fix_layers in ["all", "bottom_n", "none"]:
                                            for fix_layers in ["none"]:
                                                # for fix_layers_n in [8]:
                                                # for feature_layers in ["last", "top_n"]:
                                                # for feature_layers in ["last"]:
                                                for feature_layers in ["top_n"]:
                                                    for feature_layers_n in [1]:
                                                        # --auto_layer
                                                        for auto_layer in ["--auto_layer"]:
                                                            info = "{description}_{var}_{var1}_{var2}_{var3}".format(
                                                                description="e47",
                                                                var=dataset,
                                                                var1=ptm,
                                                                var2="latent_er",
                                                                var3=m_size)
                                                            #  --feature_layers_n {feat_l_n} --fix_layers_n {fix_l_n}
                                                            cmd = 'python3 -m utils.main --info {info} --seed {seed} ' \
                                                                  '--model {model} --area NLP --dataset {dataset} ' \
                                                                  '--buffer_size {m_size} ' \
                                                                  '--csv_log --tensorboard --lr {lr} --minibatch_size 32 ' \
                                                                  '--ptm {ptm} ' \
                                                                  '--eval_freq 1 --prob_type {p_type} --prob_all_tasks ' \
                                                                  '--beta {beta} --batch_size 32 --n_epochs {epoch} ' \
                                                                  '--fix_layers {fix_l} ' \
                                                                  '--feature_layers {feat_l} --feature_layers_n {feat_l_n} ' \
                                                                  '{auto_l}'.format(model=model,
                                                                                    beta=beta,
                                                                                    dataset=dataset,
                                                                                    seed=seed,
                                                                                    ptm=ptm,
                                                                                    lr=lr,
                                                                                    p_type=p_type,
                                                                                    epoch=epoch,
                                                                                    info=info,
                                                                                    fix_l=fix_layers,
                                                                                    # fix_l_n=fix_layers_n,
                                                                                    feat_l=feature_layers,
                                                                                    feat_l_n=feature_layers_n,
                                                                                    m_size=m_size,
                                                                                    auto_l=auto_layer)
                                                            info_list.append(info)
                                                            cmd_list.append(cmd)
    return cmd_list, info_list
