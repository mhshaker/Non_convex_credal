# Monotonicity test for set 16
import random
import numpy as np
import UncertaintyM as unc

for j in range(100):
    last_t = 0
    rp = random.uniform(0, 1)
    rp2 = random.uniform(0, 1)
    prob = np.array([[[rp, 1-rp], [rp2, 1-rp2]]])

    for i in range(50):

        rp = random.uniform(0, 1)
        prob_add = np.array([[[rp, 1-rp]]])
        prob = np.concatenate((prob,prob_add), axis=1)
        t, e, a = unc.uncertainty_set16(prob)
        if t < last_t:
            # print(prob)
            print(f"Monotonicity test failed {j} last_t {last_t} t {t}")
            break
        else:
            last_t = t
print("done")