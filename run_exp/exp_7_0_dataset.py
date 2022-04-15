"""


Author: Tong
Time: --2021
"""

dataset_map = {"seq-clinc150": "c", "seq-maven": "m", "seq-webred": "w"}


def generate_cmd():
    cmd_list = []
    info_list = []
    
    for dataset in ["seq-clinc150", "seq-maven", "seq-webred"]:
        info = "{description}_{var}".format(description="data", var=dataset_map[dataset])
        cmd = 'python3 -m analyze.dataset_distribution --info {info} --dataset {dataset}'.format(dataset=dataset,
                                                                                                 info=info)
        info_list.append(info)
        cmd_list.append(cmd)
    return cmd_list, info_list
