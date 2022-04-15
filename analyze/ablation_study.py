# Libraries
import matplotlib.pyplot as plt
import pandas as pd
from math import pi

# Set data
df = pd.DataFrame({
    'group': ['joint','pseudoCL', 'w/o replay', 'w/o pseudo', 'w/o distill'],
    'cil-acc': [95.69, 86.29, 84.0, 74, 1],
    'til-acc': [97.07, 94.71, 93.0, 92, 1],
    'iil-acc': [94.0, 90, 84.0, 83, 1],
    # 'time': [1.41, 0, 0, 0],
    # 'forward transfer': [-0.86, 0, 0, 0],
    # 'backward transfer': [-12.1, 0, 0, 0],
    # 'forgetting rate': [12.1, 0, 0, 0]
})

color = ["r", "b", "g", "y"]
group = ['joint','pseudoCL', 'w/o replay', 'w/o pseudo', 'w/o distill']

# ------- PART 1: Create background

# number of variable
categories = list(df)[1:]
N = len(categories)

# What will be the angle of each axis in the plot? (we divide the plot / number of variable)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]

# Initialise the spider plot
ax = plt.subplot(111, polar=True)

# If you want the first axis to be on top:
ax.set_theta_offset(pi / 2)
ax.set_theta_direction(-1)

# Draw one axe per variable + add labels
plt.xticks(angles[:-1], categories)

# Draw ylabels
ax.set_rlabel_position(0)
plt.yticks([70, 80, 90], ["70", "80", "90"], color="grey", size=7)
plt.ylim(60, 100)

# ------- PART 2: Add plots

# Plot each individual = each line of the data
# I don't make a loop, because plotting more than 3 groups makes the chart unreadable

# # Ind1
# values = df.loc[0].drop('group').values.flatten().tolist()
# values += values[:1]
# ax.plot(angles, values, linewidth=1, linestyle='solid', label="group A")
# # ax.fill(angles, values, 'b', alpha=0.1)
#
# # Ind2
# values = df.loc[1].drop('group').values.flatten().tolist()
# values += values[:1]
# ax.plot(angles, values, linewidth=1, linestyle='solid', label="group B")
# # ax.fill(angles, values, 'r', alpha=0.1)

# Ind3
for i, label in enumerate(group):
    values = df.loc[i].drop('group').values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=1, linestyle='solid', label=label)
    # ax.fill(angles, values, 'r', alpha=0.1)


# Add legend
plt.legend()

# Show the graph
plt.show()
