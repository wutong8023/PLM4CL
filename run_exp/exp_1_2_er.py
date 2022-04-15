"""


Author: Tong
Time: 28-04-2021
"""
map = {"seq-clinc150": "c", "seq-maven": "m", "seq-webred": "w"}


def generate_cmd():
    cmd_list = []
    info_list = []
    for model in ["er_nlp"]:
        for ptm in ["bert", "roberta", "xlnet"]:
            for dataset in ["seq-maven"]:
                for seed in range(1):
                    seed = seed * 100
                    for filter_rate in [1]:
                        for prob_l in [-1]:
                            for lr in [0.00001]:
                                for beta in [1.0]:
                                    for epoch in [20]:
                                        info = "{description}_{var}_{var1}_{var2}".format(description="e12",
                                                                                          var=map[dataset], var1=ptm,
                                                                                          var2="er")
                                        cmd = 'python3 -m utils.main --info {info} --seed {seed} ' \
                                              '--model {model} --area NLP --dataset {dataset} --buffer_size 500 ' \
                                              '--csv_log --tensorboard --lr {lr} --minibatch_size 32 --ptm {ptm} ' \
                                              '--eval_freq 1 --prob_l {prob_l} ' \
                                              '--beta {beta} --batch_size 32 --n_epochs {epoch}'.format(model=model,
                                                                                                        beta=beta,
                                                                                                        dataset=dataset,
                                                                                                        seed=seed,
                                                                                                        ptm=ptm,
                                                                                                        lr=lr,
                                                                                                        epoch=epoch,
                                                                                                        info=info,
                                                                                                        prob_l=prob_l)
                                        info_list.append(info)
                                        cmd_list.append(cmd)
    return cmd_list, info_list
