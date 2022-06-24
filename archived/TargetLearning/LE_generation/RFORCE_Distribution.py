import numpy as np
import numpy.random as npr
from numpy import linalg as LA

def R_FORCEDistribution(N, g, index = 1):
    radius = np.array([0.9, 0.7, 0.72, 1.2]) * g
    num_circles = len(radius)
    partition_out = 0.01 * np.ones([1, 1])
    partition_in = 1 - partition_out
    distance_thresold = 1.15
    if radius[num_circles - 1] > 1.55:
        per_out = partition_out * N / 2
        per_in = np.zeros([1, num_circles - 1])
        radius_in = radius[:-1]
        distance = np.absolute(radius_in - distance_thresold)
        percentage = (1 / distance) ** (2 * g)
        percentage_normalized = percentage / np.sum(percentage)
        for i in range(num_circles - 2):
            a = np.maximum(np.floor(percentage_normalized[i] * N / 2), 0.1 * N / 2)
            per_in[0][i] = a
        per_in[0][-1] = partition_in * N / 2 - np.sum(per_in)
        per = np.concatenate((per_in, per_out), axis=1)
    else:
        radius_in = radius
        distance = np.absolute(radius_in - distance_thresold)
        percentage = np.tanh(-2 * distance) + 1
        percentage_normalized = percentage / np.sum(percentage)
        for i in range(num_circles - 1):
            per[0, i] = np.floor(percentage_normalized[i] * N / 2)

        per[0, -1] = N / 2 - np.sum(per)
    count = 0
    theta = np.zeros([1, int(N / 2)])
    real_part = np.zeros([1, int(N / 2)])
    imag_part_pos = np.zeros([1, int(N / 2)])

    if (g < 1.8):
        theta_range = [0.02, np.pi / 3, np.pi / 3, 2 * np.pi / 3, 2 * np.pi / 3, np.pi - 0.02]
    else:
        theta_range = [4 * np.pi / 5, np.pi - 0.02, 2 * np.pi / 5, 4 * np.pi / 5, 0.02, 2 * np.pi / 5]
    for i in range(num_circles - 1):
        theta[0, count: count + int(per[0, i])] = np.linspace(theta_range[2 * i], theta_range[2 * i + 1],
                                                              num=int(per[0][i]))

        real_part[0, count: count + int(per[0, i])] = radius[i] * np.cos(theta[0, count: count + int(per[0, i])]);
        imag_part_pos[0, count: count + int(per[0, i])] = radius[i] * np.sin(theta[0, count: count + int(per[0, i])]);
        count = count + int(per[0][i])

    i = num_circles - 1
    if (g < 1.4):
        theta[0, count: count + int(per[0, i])] = np.linspace(0, 1.2 * np.pi / (g ** 2 * (num_circles - 1)),
                                                              num=int(per[0, i]))
    else:
        theta[0, count: count + int(per[0, i])] = np.linspace(0.4 * np.pi, 0.6 * np.pi, num=int(per[0, i]))

    real_part[0, count: count + int(per[0, i])] = radius[i] * np.cos(theta[0, count: count + int(per[0, i])]);
    imag_part_pos[0, count: count + int(per[0, i])] = radius[i] * np.sin(theta[0, count: count + int(per[0, i])]);

    count = count + int(per[0][i])

    imag_part_neg = - imag_part_pos;

    # Generate the eigenvectors matrix
    Temp = np.random.normal(0, 1, [N, N])

    Temp = Temp - np.transpose(Temp)

    [D, V] = LA.eig(Temp)

    eig_val = np.zeros([N]) * 1j

    M = np.zeros([N, N]) * 1j

    for i in range(int(N / 2)):
        eig_val[2 * i] = real_part[0, i] + 1j * imag_part_pos[0, i]
        eig_val[2 * i + 1] = real_part[0, i] + 1j * imag_part_neg[0, i]

    M = np.dot(np.dot(V, np.diag(eig_val)), LA.inv(V))
    M = np.real(M)
    per = per / (N / 2)
    return M, per, radius, theta


def main():
    N = 100
    g = 1.5
    M, per, radius, theta = R_FORCEDistribution(N=N, g=g)
    print(M.shape)

# if __name__ == "__main__":
#     main()