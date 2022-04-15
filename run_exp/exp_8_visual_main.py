"""


Author: Tong
Time: --2021
"""

dataset_map = {"seq-clinc150": "c", "seq-maven": "m", "seq-webred": "w"}
vtype = ["time", "forward_transfer", "backward_transfer", "forgetting", "mean_accuracy","all"]


def generate_cmd():
    cmd_list = []
    info_list = []
    
    for v_type in vtype:
        for setting in ["class", "task"]:
            for v_by in ["ptm", "mtd"]:
                info = "{description}_{var}".format(description="main", var=v_type)
                cmd = 'python3 -m analyze.time_ft_bt_acc_analysis --info {info} ' \
                      ' --vis_type {v_type} ' \
                      ' --vis_by {v_by} --setting {setting}'.format(info=info,
                                                                    v_by=v_by,
                                                                    v_type=v_type,
                                                                    setting=setting)
                info_list.append(info)
                cmd_list.append(cmd)
    
    return cmd_list, info_list
