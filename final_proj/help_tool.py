import itertools
import mujoco

model = mujoco.MjModel.from_xml_path('mjmodel.xml')

# Retrieve the number of bodies
num_bodies = model.nbody

# Get the names of all bodies
body_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(num_bodies)]


# Generate exclusion pairs
exclusions = list(itertools.combinations(body_names, 2))

# Create the XML exclusion tags
exclusion_tags = "\n".join([f'<exclude body1="{pair[0]}" body2="{pair[1]}"/>' for pair in exclusions])

# Print or save the generated exclusion tags
print(exclusion_tags)

# Example output for the above body names
# <exclude body1="body1" body2="body2"/>
# <exclude body1="body1" body2="body3"/>
# <exclude body1="body1" body2="body4"/>
# <exclude body1="body2" body2="body3"/>
# <exclude body1="body2" body2="body4"/>
# <exclude body1="body3" body2="body4"/>
