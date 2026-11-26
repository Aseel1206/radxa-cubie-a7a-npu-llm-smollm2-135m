import numpy as np,h5py
D='/workspace/mp/llm/calib/'
with h5py.File(D+'all.h5','w') as f:
    for n in ['x','cos','sin','mask','pk','pv']: f.create_dataset(n,data=np.load(D+f'all_{n}.npy').astype(np.float32))
print('ok')
