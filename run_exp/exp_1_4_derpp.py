"""


Author: Tong
Time: 28-04-2021
"""
map = {"seq-clinc150": "c", "seq-maven": "m", "seq-webred": "w"}


def generate_cmd():
    cmd_list = []
    info_list = []
    for model in ["derpp_nlp"]:
        for seed in range(1):
            seed = seed * 100
            for ptm in ["albert"]:
                for dataset in ["seq-maven"]:
                    for alpha in [0.5]:
                        for prob_l in [-1]:
                            for lr in [0.00001]:
                                for beta in [1.0]:
                                    for epoch in [20]:
                                        info = "{description}_{var}_{var1}_{var2}".format(description="e14",
                                                                                          var=map[dataset], var1=ptm,
                                                                                          var2="derpp")
                                        
                                        cmd = 'python3 -m utils.main --info {info} --seed {seed} ' \
                                              '--model {model} --area NLP --dataset {dataset} --buffer_size 500 ' \
                                              '--csv_log --tensorboard --lr {lr} --minibatch_size 16 --ptm {ptm} ' \
                                              '--alpha {alpha} --beta {beta} --batch_size 16 ' \
                                              '--eval_freq 1 --prob_l {prob_l} ' \
                                              '--n_epochs {epoch}'.format(model=model,
                                                                          alpha=alpha,
                                                                          beta=beta,
                                                                          seed=seed,
                                                                          prob_l=prob_l,
                                                                          dataset=dataset,
                                                                          ptm=ptm,
                                                                          lr=lr,
                                                                          epoch=epoch,
                                                                          info=info)
                                        info_list.append(info)
                                        cmd_list.append(cmd)
    
    return cmd_list, info_list
