import matplotlib.pyplot as plt;

plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt


def gamma_dist_simulation(alpha):
    K = len(alpha)
    z = np.array([np.random.gamma(alpha[i], 1) for i in range(K)])
    q = z / np.sum(z)
    return q


def polya_urn_simulation(alpha):
    alpha = list(alpha)
    balls = [str(i) for i in range(len(alpha))]
    urn = dict(zip(balls, alpha))
    num_samples = 1000
    for i in range(num_samples):
        p = np.array(list(urn.values())) / np.sum(list(urn.values()))
        draw_obj = np.random.choice(list(urn.keys()), 1, p=p)[0]
        urn[draw_obj] += 1

    q = np.array(list(urn.values())) / sum(list(urn.values()))
    return q


def stick_breaking_simulation(alpha):
    alpha = np.array(alpha)
    K = len(alpha)
    u = np.zeros(K)
    q = np.zeros(K)
    for i in range(K - 1):
        u[i] = np.random.beta(alpha[i], np.sum(alpha[i + 1:]))
        q[i] = u[i] * np.prod(1 - u[:i])
    q[-1] = 1 - np.sum(q)
    return q


def plot_samples_from_dirichlet(alpha):
    for i in range(25):
        # q = stick_breaking_simulation(alpha)
        # q = polya_urn_simulation(alpha)
        q = gamma_dist_simulation(alpha)
        # q = np.random.dirichlet(alpha)
        plt.subplot(5, 5, i + 1)
        y_pos = np.arange(len(q))
        axes = plt.gca()
        axes.set_ylim([0, 1])
        plt.bar(y_pos, q, align='center', alpha=0.5)

    plt.show()


# alpha = [0.01, 0.01, 0.01] # sparse
# alpha = [10, 2, 8]


if __name__ == "__main__":
    '''
    One can think about dirichlet distribution as a machine that creates dices, each dice represents a prob distribution
    Not all dices are fair. 
    Alpha is the parameter that tells how to create fair/unfair dices. If all alpha values are the same(symmetric) then we have
    symmetric (=fair) "way" of creating dices but this doesn't mean dices would be fair. It just means we fairly create fair/unfair (bad/good) dices.
    But again if alpha values are higher then there is a less variance in creating them. It means we create very good fair dices if alpha is big.
    But small values of alpha(<1) means we have a very bad quality control which means we create a lot of different types of unfair (for example one dice may be skewed to produce 6 more or the other one produce 2 more)
    but this bad quality is uniformly happens for all different kinds of unfair dices.
    
    if we have (10, 1, 1, 1, 1, 1) it means we create unfair dices that favor 1 with little variance. (looks like we intentionally create unfair dices but we know this happens systematically)
    but if we have (0.5, 0.1, 0.1, 0.1, 0.1, 0.1) then we favor 1 again but this is not happening systematically which means there is a high variance in the produced dices.
    
    '''
    alpha = [10, 10, 10, 10, 10, 10]
    # alpha = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    # alpha = [8, 1, 1, 1, 1, 1]
    # alpha = [0.8, 0.1, 0.1, 0.1, 0.1, 0.1]

    # alpha = [8,1,1]

    j = 0
    alpha_0 = sum(alpha)
    x1_mean = alpha[j] / alpha_0
    print("mean", x1_mean)
    variance = alpha[j] * (alpha_0 - alpha[j]) / (alpha_0 ** 2 * (alpha_0 + 1))
    print("variance", variance)
    all_pmfs = []
    for _ in range(25000):
        pmf = np.random.dirichlet(alpha)
        all_pmfs.append(pmf)

    x1_mean_exprimental = np.mean(np.stack(all_pmfs)[:, j])
    x1_var_exprimental = np.var(np.stack(all_pmfs)[:, j])
    print(x1_mean_exprimental)
    print(x1_var_exprimental)

    plot_samples_from_dirichlet(alpha)
    # q = polya_urn_simulation(alpha)
    # q = stick_breaking_simulation(alpha=[7,1,1])
    q = gamma_dist_simulation(alpha)
    print(q)
