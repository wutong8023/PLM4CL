"""


Author: Tong
Time: 28-04-2021
"""

def generate_cmd():
    cmd_list = []
    info_list = []
    for model in ["er_nlp"]:
        # for dataset in ["seq-clinc150", "seq-maven", "seq-webred"]:
        for dataset in ["seq-clinc150"]:
            for seed in range(1):
                seed = seed * 100
                for ptm in ["bert", "gpt2"]:
                    for lr in [0.0001]:
                        for beta in [1.0]:
                            for epoch in [50]:
                                for p_type in ["proto"]:
                                    if ptm == "albert": epoch = 15
                                    info = "{description}_{var}_{var1}_{var2}".format(description="e52", var=p_type,
                                                                                      var1=ptm, var2="er")
                                    cmd = 'python3 -m utils.main --info {info} --seed {seed} ' \
                                          '--model {model} --area NLP --dataset {dataset} --buffer_size 500 ' \
                                          '--csv_log --tensorboard --lr {lr} --minibatch_size 32 --ptm {ptm} ' \
                                          '--eval_freq 1 --prob_type {p_type} ' \
                                          '--beta {beta} --batch_size 32 --n_epochs {epoch}'.format(model=model,
                                                                                                    beta=beta,
                                                                                                    dataset=dataset,
                                                                                                    seed=seed,
                                                                                                    ptm=ptm,
                                                                                                    lr=lr,
                                                                                                    p_type=p_type,
                                                                                                    epoch=epoch,
                                                                                                    info=info)
                                    info_list.append(info)
                                    cmd_list.append(cmd)

                        
                                
    return cmd_list, info_list
