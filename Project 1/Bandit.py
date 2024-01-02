import numpy as np
import matplotlib.pyplot as plt
import random

k = 10  #no. of bandits
bandits = list(range(0, k))
print(bandits)

bandit_problems = 2000 #no. of bandit problems
trials = 1000 #no. of attempts per problem
q_true = np.random.normal(0, 1, k)  #The true value q⇤(a)
q_actual = np.random.normal(q_true, 1)   #The actual value q⇤(a)
print(q_actual)
Q_initial = np.zeros((1, 10), int)
print(Q_initial)
N_A = np.zeros((1, 10))  #number of turns of each arm
N = 0
Reward_A = np.zeros(trials + 1)
Avg_Reward_A = np.zeros((1, 10))  #number of turns of each arm
Q_current_reward = np.zeros(trials + 1)
Q_current_reward_matrix = np.zeros(trials + 1)
epsilon = [0, 0.01, 0.1]
for e in epsilon:
    if e == 0:
        for __ in range(2000):
            #q_actual = np.random.normal(q_true, 1)
            #A = np.argmax(q_actual)
            q_true = np.random.normal(0, 1, k)
            Avg_Reward_A = np.zeros((1, 10))
            N_A = np.zeros((1, 10))
            N = 0
            for _ in range(1000):
                q_actual = np.random.normal(q_true, 1)
                A = np.argmax(Avg_Reward_A)
                Reward_A[N] = q_actual[A]

                N_A[0, A] += 1
                Avg_Reward_A[0, A] = Avg_Reward_A[0, A] + ((1 / N_A[0, A]) * (Reward_A[N] - Avg_Reward_A[0, A]))
                N += 1
                #Q_initial[0, A] = Avg_Reward_A[0, A]


                #Q_current_reward_matrix = np.append(Q_current_reward_matrix, Q_current_reward)
            #print("1000 Done")
            #print("Reward : ", Reward_A)
            Q_current_reward = Reward_A
            #print("Reward_A = ", Reward_A)
            Q_current_reward_matrix += Q_current_reward
        print(Q_current_reward_matrix/2000)
        print(Q_current_reward)
        plt.plot((Q_current_reward_matrix/2000), linewidth=1, color='r', label='epsilon =0')
        plt.xlabel('Steps')
        plt.ylabel('Average reward')
        plt.legend()
        plt.savefig('results.png', dpi=300)
        plt.show()
        print("Done")
    else:
        range_prob = 1 / e
        Q_current_reward = np.zeros(trials + 1)
        Q_current_reward_matrix = np.zeros(trials + 1)
        for __ in range(2000):
            q_true = np.random.normal(0, 1, k)
            Avg_Reward_A = np.zeros((1, 10))
            N_A = np.zeros((1, 10))
            N = 0
            for _ in range(1000):
                arr = np.random.randint(0, range_prob, 1)
                if arr == 1:
                    q_actual = np.random.normal(q_true, 1)
                    A = np.random.randint(0, k)
                    Reward_A[N] = q_actual[A]
                    N_A[0, A] += 1
                    Avg_Reward_A[0, A] = Avg_Reward_A[0, A] + ((1 / N_A[0, A]) * (Reward_A[N] - Avg_Reward_A[0, A]))
                    N += 1
                    #print(_)
                else:
                    q_actual = np.random.normal(q_true, 1)
                    A = np.argmax(Avg_Reward_A)
                    Reward_A[N] = q_actual[A]
                    N_A[0, A] += 1
                    Avg_Reward_A[0, A] = Avg_Reward_A[0, A] + ((1 / N_A[0, A]) * (Reward_A[N] - Avg_Reward_A[0, A]))
                    N += 1
                #print(_)
            Q_current_reward = Reward_A
            Q_current_reward_matrix += Q_current_reward
    print(Q_current_reward_matrix/2000)
    print(Q_current_reward)
    plt.plot((Q_current_reward_matrix/2000), linewidth=2, color='k', label='epsilon =' + str(e))
    plt.xlabel('Steps')
    plt.ylabel('Average reward')
    plt.legend()
    plt.savefig('results.png', dpi=300)
    plt.show()








