"""


Author: Tong
Time: --2021
"""

dataset_map = {"seq-clinc150": "c", "seq-maven": "m", "seq-webred": "w"}
visual_type = ["model_layer_mtd_final", "model_layer_mtd_mean_std",
                "model_task_layer", "model_layer_bs_mean_std_fine",
                "model_layer_bs_mean_std_coarse", "all_model_layer_bs_mean_std_coarse",
                "all_model_layer_bs_mean_std"]
    

def generate_cmd():
    cmd_list = []
    info_list = []
    
    for dataset in ["seq-clinc150"]:
        for p_type in ["proto"]:
            for mtd in ["er", "derpp"]:
                for v_type in visual_type:
                    for setting in ["class", "task"]:
                        for v_by in ["ptm"]:
                            info = "{description}_{var}".format(description="prob", var=v_type)
                            cmd = 'python3 -m analyze.layer_probing --info {info} --dataset {dataset} ' \
                                  '--vis-type model_layer_mtd_final --vis_type {v_type} --prob_mtd {mtd} ' \
                                  '--prob_type {p_type} --vis_by {v_by} --setting {setting}'.format(dataset=dataset,
                                                                                                    info=info,
                                                                                                    p_type=p_type,
                                                                                                    v_by=v_by,
                                                                                                    mtd=mtd,
                                                                                                    v_type=v_type,
                                                                                                    setting=setting)
                            info_list.append(info)
                            cmd_list.append(cmd)
    
    # for dataset in ["seq-webred"]:
    #     for p_type in ["proto"]:
    #         for mtd in ["er", "derpp"]:
    #             for v_type in ["model_layer_mtd_final", "model_layer_mtd_mean_std", "model_task_layer"]:
    #                 for setting in ["class", "task"]:
    #                     for v_by in ["ptm"]:
    #                         info = "{description}_{var}".format(description="prob", var=v_type)
    #                         cmd = 'python3 -m analyze.layer_probing --info {info} --dataset {dataset} ' \
    #                               '--vis-type model_layer_mtd_final --vis_type {v_type} --prob_mtd {mtd} ' \
    #                               '--prob_type {p_type} --vis_by {v_by} --setting {setting}'.format(dataset=dataset,
    #                                                                                                 info=info,
    #                                                                                                 p_type=p_type,
    #                                                                                                 v_by=v_by,
    #                                                                                                 mtd=mtd,
    #                                                                                                 v_type=v_type,
    #                                                                                                 setting=setting)
    #                         info_list.append(info)
    #                         cmd_list.append(cmd)
    #
    #
    #
    
    return cmd_list, info_list
