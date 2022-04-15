"""


Author: Tong
Time: 28-04-2021
"""
map = {"seq-clinc150": "c", "seq-maven": "m", "seq-webred": "w"}


def generate_cmd():
    cmd_list = []
    info_list = []
    for model in ["derpp_nlp"]:
        for dataset in ["seq-clinc150"]:
            for seed in range(1):
                seed = seed * 100
                for ptm in ["bert", "xlnet", "albert", "roberta", "gpt2"]:
                    for alpha in [0.5]:
                        for lr in [0.0001]:
                            for beta in [1.0]:
                                for epoch in [50]:
                                    for p_type in ["proto", "final"]:
                                        if ptm == "albert":
                                            epoch = 15
                                        info = "{description}_{var}_{var1}_{var2}".format(description="e54",
                                                                                          var=map[dataset], var1=ptm,
                                                                                          var2="derpp")
    
                                        cmd = 'python3 -m utils.main --info {info} --seed {seed} ' \
                                              '--model {model} --area NLP --dataset {dataset} --buffer_size 500 ' \
                                              '--csv_log --tensorboard --lr {lr} --minibatch_size 32 --ptm {ptm} ' \
                                              '--alpha {alpha} --beta {beta} --batch_size 32 ' \
                                              '--eval_freq 1 --prob_type {p_type} ' \
                                              '--n_epochs {epoch}'.format(model=model,
                                                                          alpha=alpha,
                                                                          beta=beta,
                                                                          seed=seed,
                                                                          dataset=dataset,
                                                                          ptm=ptm,
                                                                          lr=lr,
                                                                          p_type=p_type,
                                                                          epoch=epoch,
                                                                          info=info)
                                        info_list.append(info)
                                        cmd_list.append(cmd)
    
    return cmd_list, info_list
