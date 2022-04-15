"""


Author: Tong
Time: --2021
"""

dataset_map = {"seq-clinc150": "c", "seq-maven": "m", "seq-webred": "w"}


def generate_cmd():
    cmd_list = []
    info_list = []
    
    for dataset in ["seq-clinc150"]:
        for ptm in ["bert"]:
            info = "{description}_{var}_{var1}_{var2}".format(description="e10",
                                                              var=dataset_map[dataset], var1=ptm,
                                                              var2="joint")
            cmd = 'python3 -m analyze.backbone_flops --info {info} --dataset {dataset} ' \
                  '--ptm {ptm} '.format(dataset=dataset,
                                        ptm=ptm,
                                        info=info)
            info_list.append(info)
            cmd_list.append(cmd)
    return cmd_list, info_list
