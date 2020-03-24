import numpy as np

dataset_size = 10000
z_dims = 2
mean = 0
sigma = 1

# toy model 1
np.random.seed(1)
factors = np.random.normal(loc=mean, scale=sigma, size=(dataset_size, z_dims))
representations = factors

representations[:, 1] = 1/np.sqrt(2) * (factors[:, 0] + factors[:, 1])
representations = representations + np.random.normal(loc=0, scale=0.01, size=(dataset_size, z_dims))

np.savetxt('toy_dataset_factors1', factors, fmt='%.6f')
np.savetxt('toy_dataset_representations1', representations, fmt='%.6f')

# toy model 2
np.random.seed(1)
factors = np.random.normal(loc=mean, scale=sigma, size=(dataset_size, z_dims))
representations = np.zeros((dataset_size, 8))

representations[:, 0] = 1/np.sqrt(5) * (factors[:, 0] + 2*factors[:, 1])
representations[:, 1] = 1/np.sqrt(5) * (factors[:, 0] - 2*factors[:, 1])
representations[:, 2] = 1/np.sqrt(5) * (2*factors[:, 0] + factors[:, 1])
representations[:, 3] = 1/np.sqrt(5) * (2*factors[:, 0] - factors[:, 1])
representations[:, 4] = 1/np.sqrt(2) * (factors[:, 0] + factors[:, 1])
representations[:, 5] = -1/np.sqrt(2) * (factors[:, 0] + factors[:, 1])
representations[:, 6] = 1/np.sqrt(2) * (factors[:, 0] - factors[:, 1])
representations[:, 7] = -1/np.sqrt(2) * (factors[:, 0] - factors[:, 1])

representations = representations + np.random.normal(loc=0, scale=0.01, size=representations.shape)

np.savetxt('toy_dataset_factors2', factors, fmt='%.6f')
np.savetxt('toy_dataset_representations2', representations, fmt='%.6f')