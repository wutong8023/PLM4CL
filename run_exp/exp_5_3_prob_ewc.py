"""


Author: Tong
Time: --2021
"""


def generate_cmd():
    cmd_list = []
    info_list = []
    
    for model in ["ewc_on_nlp"]:
        # for dataset in ["seq-clinc150", "seq-maven", "seq-webred"]:
        for dataset in ["seq-clinc150"]:
            for seed in range(1):
                seed = seed * 100
                for ptm in ["bert", "xlnet", "albert", "roberta", "gpt2"]:
                    for filter_rate in [1]:
                        for lr in [0.0001]:
                            for epoch in [50]:
                                for e_lambda in [10]:
                                    for p_type in ["final"]:
                                        if ptm == "albert": epoch = 15
                                        info = "{description}_{var}_{var1}_{var2}".format(description="e53",
                                                                                          var=p_type,
                                                                                          var1=ptm, var2="ewc_on")
                                        cmd = 'python3 -m utils.main --info {info} --seed {seed} ' \
                                              '--model {model} --area NLP --dataset {dataset} ' \
                                              '--csv_log --tensorboard --lr {lr} --ptm {ptm} ' \
                                              '--eval_freq 1 --prob_type {p_type} --filter_rate {filter_rate} ' \
                                              '--batch_size 32 --n_epochs {epoch} ' \
                                              '--e_lambda {e_lambda} --gamma 1.0'.format(model=model,
                                                                                         dataset=dataset,
                                                                                         seed=seed,
                                                                                         ptm=ptm,
                                                                                         info=info,
                                                                                         filter_rate=filter_rate,
                                                                                         lr=lr,
                                                                                         epoch=epoch,
                                                                                         p_type=p_type,
                                                                                         e_lambda=e_lambda)
                                        info_list.append(info)
                                        cmd_list.append(cmd)
                        
    
    return cmd_list, info_list
