import mujoco
import time
import numpy as np
import csv
import itertools
import matplotlib.pyplot as plt
import mujoco.viewer
import seaborn as sns


# Load MuJoCo model
model = mujoco.MjModel.from_xml_path('task1.xml')
data = mujoco.MjData(model)

data.qvel[:] = np.zeros_like(data.qvel)
data.qacc[:] = np.zeros_like(data.qacc)

# Get joint information
joint_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) for joint_id in range(model.njnt)]
joint_ranges = [model.jnt_range[joint_id] for joint_id in range(model.njnt)]
num_steps = 12 # Number of steps within each joint range

mujoco.mj_step(model, data)
# Generate all possible configurations within joint limits
joint_configs = itertools.product(
    *[np.linspace(joint_range[0], joint_range[1], num=num_steps) for joint_range in joint_ranges])

results = []


# Iterate over all joint configurations
for config in joint_configs:
    data.qpos[:] = config
    mujoco.mj_forward(model, data)
    if data.ncon == 0:
        mujoco.mj_inverse(model, data)
        torques = data.qfrc_inverse
        results.append(list(config) + list(torques))
    else:
        print(f"Collision detected for configuration: {config}")

# Save results to CSV
with open('results.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    header = joint_names + [f'torque_{name}' for name in joint_names]
    writer.writerow(header)
    writer.writerows(results)

# Plot results
# Read the data back from CSV
data = np.genfromtxt('results.csv', delimiter=',', names=True)

# Prepare data for plotting
plot_data = {name: data[f'torque_{name}'] for name in joint_names}

# Create violin plot
plt.figure(figsize=(10, 6))
sns.violinplot(data=list(plot_data.values()))
plt.xticks(range(len(joint_names)), joint_names)
plt.xlabel('Joint Name')
plt.ylabel('Torque')
plt.title('Torque Distribution Across Joints')
plt.show()




