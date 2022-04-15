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

import chart_studio.plotly as py
import chart_studio.plotly.figure_factory as ff
import chart_studio.plotly.graph_objs as go

data = [['', 'Emma', 'Isabella', 'Ava', 'Olivia', 'Sophia', 'row-sum'],
        ['Emma', 16, 3, 28, 0, 18, 65],
        ['Isabella', 18, 0, 12, 5, 29, 64],
        ['Ava', 9, 11, 17, 27, 0, 64],
        ['Olivia', 19, 0, 31, 11, 12, 73],
        ['Sophia', 23, 17, 10, 0, 34, 84]]

table = ff.create_table(data, index=True)
py.iplot(table, filename='Data-Table')