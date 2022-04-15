"""
epoch selection

Author: Tong
Time: 25-04-2021
"""

map = {"seq-clinc150": "c", "seq-maven": "m", "seq-webred": "w"}


def generate_cmd():
    cmd_list = []
    info_list = []
    
    for model in ["vanilla_nlp"]:
        for ptm in ["albert", "bert", "roberta", "xlnet", "gpt2"]:
            for dataset in ["seq-webred", "seq-maven"]:
                for seed in range(1):
                    seed = seed * 100
                    for filter_rate in [1]:
                        for prob_l in [-1]:
                            for lr in [0.00001]:
                                for epoch in [20]:
                                    info = "{description}_{var}_{var3}_{var2}".format(description="e11",
                                                                                      var=map[dataset],
                                                                                      var3=ptm, var2="vnla")
                                    cmd = 'python3 -m utils.main --info {info} --seed {seed} ' \
                                          '--model {model} --area NLP --dataset {dataset} ' \
                                          '--csv_log --tensorboard --lr {lr} --ptm {ptm} ' \
                                          '--eval_freq 1 --prob_l {prob_l} --filter_rate {filter_rate} ' \
                                          ' --batch_size 32 --n_epochs {epoch}'.format(model=model,
                                                                                       dataset=dataset,
                                                                                       seed=seed,
                                                                                       ptm=ptm,
                                                                                       info=info,
                                                                                       filter_rate=filter_rate,
                                                                                       prob_l=prob_l,
                                                                                       lr=lr,
                                                                                       epoch=epoch)
                                    info_list.append(info)
                                    cmd_list.append(cmd)

    return cmd_list, info_list
