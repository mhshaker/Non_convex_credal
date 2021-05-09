import numpy as np
import UncertaintyM as unc
from scipy.stats.contingency import margins



# Generate joint and marginal distributions

prob_joint_array = []
prob_m1_array = []
prob_m2_array = []

for j in range(5000): # data range
    prob_joint = []
    prob_m1 = []
    prob_m2 = []

    for i in range(40): # model range
        joint = np.random.randint(0,100, size=(4, 4))
        joint = joint / joint.sum()
        m1, m2 = margins(joint)
        joint = np.reshape(joint, (1,-1))
        m1 = np.reshape(m1, (1,-1))
        m2 = np.reshape(m2, (1,-1))
        prob_joint.append(joint)
        prob_m1.append(m1)
        prob_m2.append(m2)

    prob_joint = np.array(prob_joint)
    prob_m1 = np.array(prob_m1)
    prob_m2 = np.array(prob_m2)

    prob_joint = np.reshape(prob_joint, (prob_joint.shape[0],prob_joint.shape[2]))
    prob_m1 = np.reshape(prob_m1, (prob_m1.shape[0],prob_m1.shape[2]))
    prob_m2 = np.reshape(prob_m2, (prob_m2.shape[0],prob_m2.shape[2]))

    prob_joint_array.append(prob_joint)
    prob_m1_array.append(prob_m1)
    prob_m2_array.append(prob_m2)

prob1 = np.array(prob_joint_array)
prob2 = np.array(prob_m1_array)
prob3 = np.array(prob_m2_array)


# calculate Uncertainty 
t1, e1, a1 = unc.uncertainty_ent(prob1)
t2, e2, a2 = unc.uncertainty_ent(prob2)
t3, e3, a3 = unc.uncertainty_ent(prob3)

# check for sub-additivity
unc = t1
unc_add = t2 + t3 

if (unc <= unc_add).all():
    print("test Passed")
else:
    print("test Failed")

exit()
# show the distributions that violate the sub-additivity axiom
for i, (u, uu) in enumerate(zip(unc,unc_add)):
    if u > uu:
        print("joint\n",prob1[i])
        print("m1\n",prob2[i])
        print("m2\n",prob3[i])
        print("joint unc ", u)
        print("marginal unc ", uu)

