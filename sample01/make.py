import numpy as np

o = np.random.normal(0, 3, (100, 10, 3)).astype(np.float32)
np.save("sample.npy", o)
