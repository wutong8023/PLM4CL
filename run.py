'''
fine tuning

Author: Tong
Time: 10-02-2020
'''

import os
import time
from run_exp import \
    exp_1_1_vanilla, exp_1_0_joint, exp_1_2_er, exp_1_3_ewc, exp_1_4_derpp, exp_1_5_hat, \
    exp_2_0_joint, exp_2_2_size_er, exp_2_3_ewc_lmd, exp_2_4_size_derpp, \
    exp_4_0_prob_task_joint, exp_4_1_prob_task_vanilla, \
    exp_4_2_prob_task_er, exp_4_3_prob_task_ewc, exp_4_4_prob_task_derpp, exp_4_5_prob_task_hat, exp_4_6_prob_task_pseudoCL,\
    exp_4_7_prob_task_latent_er, exp_4_8_prob_task_distill, exp_4_9_prob_task_auto_er, exp_4_10_prob_task_der, \
    exp_4_11_prob_task_auto_vanilla, exp_4_11_1_prob_task_auto_vanilla, exp_4_11_3_prob_task_auto_vanilla, \
    exp_4_12_prob_task_vanilla_inst,\
    exp_5_0_prob_joint, exp_5_1_prob_vanilla, exp_5_2_prob_er, exp_5_3_prob_ewc, exp_5_4_prob_derpp, \
    exp_5_visual_prob, \
    exp_6_0_comput, \
    exp_7_0_dataset, \
    exp_8_visual_main


def batch_submit_multi_jobs(cmd_list, info_list, platform: str, split_num: int = 4, partition="g"):
    assert len(cmd_list) == len(info_list)
    
    content = []
    file_name = "./job_base_{pltf}.sh".format(pltf=platform)
    file_out = "./job_{pltf}.sh".format(pltf=platform)
    
    cmd_list_frac = []
    info_list_frac = []
    
    flag_idx = 0
    while flag_idx < len(cmd_list):
        if (flag_idx + split_num) <= len(cmd_list):
            next_flag_idx = flag_idx + split_num
        else:
            next_flag_idx = len(cmd_list)
        
        sub_cmd_list = cmd_list[flag_idx:next_flag_idx:]
        sub_info_list = info_list[flag_idx:next_flag_idx:]
        
        cmd_list_frac.append(sub_cmd_list)
        info_list_frac.append(sub_info_list)
        
        flag_idx = next_flag_idx
    
    with open(file_name) as in_file:
        for line in in_file:
            content.append(line)
    for i, sub_cmd_list in enumerate(cmd_list_frac):
        with open(file_out, "w") as out_file:
            
            # job_name
            job_name = "__".join(info_list_frac[i])
            print("- JOB NAME: ", job_name)
            if platform == "group":
                _info = "#SBATCH -J {job_name}\n".format(job_name=job_name)
                content[21] = _info
                # SBATCH -o log/fs2s-iwslt-%J.out
                # SBATCH -e log/fs2s-iwslt-%J.err
                _out_file = "#SBATCH -o log/%J-{job_name}.out\n".format(job_name=job_name)
                content[15] = _out_file
                _err_file = "#SBATCH -e log/%J-{job_name}.err\n".format(job_name=job_name)
                content[16] = _err_file
            
            else:
                _partition = "#SBATCH --partition={var}\n".format(var=partition)
                content[2] = _partition
                _info = "#SBATCH --job-name={job_name}\n".format(job_name=job_name)
                content[3] = _info
                
                # SBATCH --output=log/fs2s-iwslt-%j.out
                # SBATCH --error=log/fs2s-iwslt-%j.err
                _out_file = "#SBATCH --output=log/%j-{job_name}.out\n".format(job_name=job_name)
                content[4] = _out_file
                _err_file = "#SBATCH --error=log/%j-{job_name}.err\n".format(job_name=job_name)
                content[5] = _err_file
            
            for line in content:
                out_file.write(line)
            
            # command
            if platform == "m3":
                pltf = "  --pltf {pltf}".format(pltf="m")
            else:
                pltf = "  --pltf {pltf}".format(pltf="gp")
            for cmd in sub_cmd_list:
                cmd = cmd + pltf
                out_file.write(cmd)
                out_file.write("\n\n")
        cmd = "sbatch job_{pltf}.sh".format(pltf=platform)
        os.system(cmd)


def batch_run_interactive(cmd_list: [str], order=1):
    # print(cmd_list)
    for i in cmd_list[::order]:
        print(i)
    for i in cmd_list[::order]:
        try:
            i = i + "  --pltf m"
            os.system(i)
            time.sleep(10)
            print(i)
        except:
            print(i, " failed!")


# cancel slurm jobs
def batch_cancel(job_start: int, num: int, platform: str):
    for i in range(job_start, job_start + num):
        if platform == "group":
            cmd = "scancel -v {i}".format(i=i)
        else:
            cmd = "scancel {i}".format(i=i)
        os.system(cmd)


if __name__ == '__main__':
    # main experiment
    set1_0 = [exp_1_0_joint]
    set1_1 = [exp_1_1_vanilla]
    set1_2 = [exp_1_2_er]
    set1_3 = [exp_1_3_ewc]
    set1_4 = [exp_1_4_derpp]
    set1_5 = [exp_1_5_hat]
    # set1_5 = [exp_1_5_hat]
    set1 = [exp_1_0_joint, exp_1_1_vanilla, exp_1_2_er, exp_1_3_ewc]
    
    
    # hyper-parameter search
    set2 = [exp_2_0_joint, exp_2_2_size_er, exp_2_3_ewc_lmd]
    set2_0 = [exp_2_0_joint]
    set2_2 = [exp_2_2_size_er]
    set2_3 = [exp_2_3_ewc_lmd]
    set2_4 = [exp_2_4_size_derpp]
    
    # probing task experiment
    set4 = [exp_4_0_prob_task_joint, exp_4_5_prob_task_hat]
    set4_all = [exp_4_3_prob_task_ewc, exp_4_0_prob_task_joint, exp_4_1_prob_task_vanilla, exp_4_2_prob_task_er, exp_4_4_prob_task_derpp]
    set4_0 = [exp_4_0_prob_task_joint]
    set4_1 = [exp_4_1_prob_task_vanilla]
    set4_2 = [exp_4_2_prob_task_er]
    set4_4 = [exp_4_4_prob_task_derpp]
    set4_3 = [exp_4_3_prob_task_ewc]
    set4_5 = [exp_4_5_prob_task_hat]
    set4_6 = [exp_4_6_prob_task_pseudoCL]
    set4_7 = [exp_4_7_prob_task_latent_er]
    set4_8 = [exp_4_8_prob_task_distill]
    set4_9 = [exp_4_9_prob_task_auto_er]
    set4_10 = [exp_4_10_prob_task_der]
    set4_11 = [exp_4_11_prob_task_auto_vanilla]
    set4_11_1 = [exp_4_11_1_prob_task_auto_vanilla]
    set4_11_3 = [exp_4_11_3_prob_task_auto_vanilla]
    set4_12 = [exp_4_12_prob_task_vanilla_inst]
    
    set4_14 = [exp_4_2_prob_task_er,exp_4_1_prob_task_vanilla]
    set4_78 = [exp_4_7_prob_task_latent_er, exp_4_8_prob_task_distill]
    
    # probing experiment
    set5_v = [exp_5_visual_prob]
    set5 = [exp_5_0_prob_joint, exp_5_1_prob_vanilla, exp_5_2_prob_er, exp_5_3_prob_ewc, exp_5_4_prob_derpp]
    set5_0 = [exp_5_0_prob_joint]
    set5_1 = [exp_5_1_prob_vanilla]
    set5_2 = [exp_5_3_prob_ewc]
    set5_3 = [exp_5_4_prob_derpp]
    
    # computation analysis
    set6 = [exp_6_0_comput]
    
    # dataset analysis
    set7 = [exp_7_0_dataset]
    
    # main result analysis
    set8 = [exp_8_visual_main]
    
    for exp in set5_v:
        # select experiment to do
        cmd_list, info_list = exp.generate_cmd()
        
        # cmd: select the running platform and the corresponding shell template
        platform = {0: "m3", 1: "group"}
        pltf = 0
        
        # cmd: running in an interactive session
        batch_run_interactive(cmd_list, order=1)
        
        # cmd: submit batch jobs for multi jobs
        # optional partition for m3: dgx , m3g, m3h, m3e, m3t
        # batch_submit_multi_jobs(cmd_list, info_list, platform[pltf], split_num=20, partition="m3t")
    
    # cmd: cancel jobs
    # batch_cancel(10750, 200, platform=platform[pltf])
