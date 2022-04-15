#!/usr/bin/env python
#
# Copyright the CoLL team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
# Intro: 
# Author: Tongtong Wu
# Time: Oct 17, 2021
"""

import os
from typing import List


def extract_table(data_list: List, layer: int):
    data = [data_list[1 + i * 12 + layer] for i in range(24)]
    data_str = "\t".join(data)
    float_data = [float(i) for i in data]
    return data_str, float_data


if __name__ == '__main__':
    # load data
    file_name = "free.csv"
    table_data = {}
    
    with open(file_name, "r") as file_in:
        for line in file_in:
            t_data = line.split("\t")
            model_name = t_data[0]
            table_data[model_name] = t_data
    print(table_data)
    
    # target_
    model = "albert"
    layer =10
    
    data_str, float_data = extract_table(table_data[model], layer)
    
    print(data_str)
    
    #
    pass
