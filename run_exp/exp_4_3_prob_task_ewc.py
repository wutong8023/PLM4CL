"""


Author: Tong
Time: --2021
"""


def generate_cmd():
    cmd_list = []
    info_list = []
    
    for model in ["ewc_on_nlp"]:
        # "seq-clinc150", "seq_maven", "seq_webred"
        for dataset in ["seq-clinc150"]:
            for seed in range(1):
                seed = seed * 100
                # "bert", "gpt2", "albert", "roberta", "xlnet"
                for ptm in ["bert"]:
                    for filter_rate in [1]:
                        for lr in [0.00001]:
                            for epoch in [20]:
                                # if whole_parameter, than regularize all parameters
                                for whole_parameter in ["--whole_parameter"]:
                                    for e_lambda in [1000000]:
                                        for p_type in ["proto"]:
                                            info = "{description}_{var}_{var1}_{var2}".format(description="e43",
                                                                                              var=p_type,
                                                                                              var1=ptm, var2="ewc_on")
                                            
                                            cmd = 'python3 -m utils.main --info {info} --seed {seed} ' \
                                                  '--model {model} --area NLP --dataset {dataset} ' \
                                                  '--csv_log --tensorboard --lr {lr} --ptm {ptm} ' \
                                                  '--eval_freq 1 --prob_type {p_type} --filter_rate {filter_rate} ' \
                                                  '--batch_size 32 {wp} --n_epochs {epoch} --prob_all_tasks ' \
                                                  '--e_lambda {e_lambda} --gamma 0.2'.format(model=model,
                                                                                             dataset=dataset,
                                                                                             seed=seed,
                                                                                             ptm=ptm,
                                                                                             info=info,
                                                                                             filter_rate=filter_rate,
                                                                                             lr=lr,
                                                                                             wp=whole_parameter,
                                                                                             epoch=epoch,
                                                                                             p_type=p_type,
                                                                                             e_lambda=e_lambda)
                                            info_list.append(info)
                                            cmd_list.append(cmd)
                                            
    return cmd_list, info_list
