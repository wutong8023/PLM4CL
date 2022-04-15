"""


Author: Tong
Time: --2021
"""

map = {"seq-clinc150": "c", "seq-maven": "m", "seq-webred": "w"}


def generate_cmd():
    cmd_list = []
    info_list = []
    
    for model in ["joint_nlp"]:
        for dataset in ["seq-clinc150"]:
            # for dataset in ["seq-clinc150"]:
            for seed in range(1):
                seed = seed * 100
                for ptm in ["gpt2", "bert", "xlnet"]:
                    for filter_rate in [1]:
                        for prob_l in [-1]:
                            for lr in [0.00001]:
                                for epoch in [30]:
                                    info = "{description}_{var}_{var1}_{var2}".format(description="e20",
                                                                                      var=map[dataset], var1=ptm,
                                                                                      var2="joint")
                                    cmd = 'python3 -m utils.main --info {info} --seed {seed} ' \
                                          '--model {model} --area NLP --dataset {dataset} ' \
                                          '--csv_log --tensorboard --lr {lr} --ptm {ptm} --increment_joint ' \
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
