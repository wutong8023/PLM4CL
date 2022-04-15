"""


Author: Tong
Time: 28-04-2021
"""
map = {"seq-clinc150": "c", "seq-maven": "m", "seq-webred": "w"}


def generate_cmd():
    cmd_list = []
    info_list = []
    for model in ["der_nlp"]:
        #  "seq-webred", "seq-maven"
        for dataset in ["seq-maven"]:
            for seed in range(1):
                seed = seed * 100
                for ptm in ["albert"]:
                    for alpha in [100]:
                        for lr in [0.00001]:
                            for m_size in [500]:
                                for epoch in [20]:
                                    for fix_layers in ["none"]:
                                        for fix_layers_n in [0]:
                                            # for feature_layers in ["last", "top_n"]:
                                            for feature_layers in ["top_n"]:
                                                for feature_layers_n in [1]:
                                                    for p_type in ["proto"]:
                                                        info = "{description}_{var1}_{var2}_{var3}".format(
                                                            description="e410",
                                                            var=p_type,
                                                            var1=ptm,
                                                            var2="der",
                                                            var3=m_size)
                                                        
                                                        cmd = 'python3 -m utils.main --info {info} --seed {seed} ' \
                                                              '--model {model} --area NLP --dataset {dataset} ' \
                                                              '--buffer_size {m_size} ' \
                                                              '--csv_log --tensorboard --lr {lr} --minibatch_size 16 --ptm {ptm} ' \
                                                              '--alpha {alpha} --batch_size 16 ' \
                                                              '--eval_freq 1 --prob_type {p_type} --prob_all_tasks ' \
                                                              '--n_epochs {epoch} ' \
                                                              '--feature_layers {feature_l} ' \
                                                              '--feature_layers_n {feature_l_n} ' \
                                                              '--fix_layers {fix_l} ' \
                                                              '--fix_layers_n {fix_l_n}'.format(model=model,
                                                                                          alpha=alpha,
                                                                                          seed=seed,
                                                                                          dataset=dataset,
                                                                                          ptm=ptm,
                                                                                          lr=lr,
                                                                                          m_size=m_size,
                                                                                          p_type=p_type,
                                                                                          epoch=epoch,
                                                                                          info=info,
                                                                                                feature_l=feature_layers,
                                                                                                feature_l_n=feature_layers_n,
                                                                                                fix_l=fix_layers,
                                                                                                fix_l_n=fix_layers_n)
                                                        info_list.append(info)
                                                        cmd_list.append(cmd)
    
    return cmd_list, info_list
