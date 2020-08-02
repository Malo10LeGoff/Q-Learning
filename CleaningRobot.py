import gym
import matplotlib as matplotlib
import matplotlib.pyplot as plt
#%matplotlib inline
import numpy as np
import random as rd
import collections


env = gym.make('gym_pallet:pallet-v0')
## on définit ici l'environnment avec le nombre de cases, d'états, les règles (quand est ce qu'on perd), ...

## hyperparamètres

discount_factor = 1.0  # grandeur représentant combien le modèle se soucie des récompenses éloignées dans le temps
alpha = 0.6  # taux d'apprentissage
epsilon = 0.1  # grandeur représentant le compromis entre l'exploration et l'exploitation
EPSILON = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
ALPHA = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
DISCOUNT_FACTOR = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]

MAX_EPSILON = 1.0
MIN_EPSILON = 0.01
DECAY_RATE = 0.0005  # taux de décroissance d'espilon

state_size = 1000000
action_size = 125


## Caracteristiques du problème

def maxListe(L):
    """
    Retourne le maximum d'une liste
    """
    a = L[0]
    for i in range(0, len(L)):
        if L[i] > a:
            a = L[i]
    return a


def Converter(action):
    """
    Convertit une action dictionnaire en liste pour pouvoir la stocker dans matrice_action
    """
    M = []
    M.append(action['ori'])
    M.append(action['pos_x'])
    M.append(action['pos_y'])
    return M


def DeConverter(L):
    """
    Convertit la liste correspondant à une action en dictionnaire correspondant à l'action originale
    """
    action = collections.OrderedDict()
    action['ori'] = L[0]
    action['pos_x'] = L[1]
    action['pos_y'] = L[2]
    return action


def action_choice(Q, Matrice_état, Matrice_action, espilon, state):
    """
    Choisit l'action à effectuer
    """
    x = rd.random()  # nombre random entre 0 et 1
    if x < epsilon:  # Exploration : on choisit une action au hasard

        action = env.action_space.sample()
        Action = Converter(action)

        if Action not in Matrice_action:
            Matrice_action.append(Action)

    else:  # on choisit l'action pour laquelle Q[index_state][action] est le plus grand avec np.argmax

        index_state = Matrice_état.index(state)

        if np.argmax(Q[index_state]) != 0:

            index_best_action = np.argmax(Q[
                                              index_state])  # np.argmax renvoie l'indice de la colonne du plus grand élément de la ligne index_state

            Action = Matrice_action[
                index_best_action]  # on retrouve l'action qui donne la plus grande récompense en la cherchant dans la liste d'action

            action = DeConverter(Action)  # on la reconvertit en dictionnaire pour que la fonction step la reconaisse

        else:  ## si on ne connait pas encore l'état alors on choisit une action au hasard

            action = env.action_space.sample()
            Action = Converter(action)
            if Action not in Matrice_action:
                Matrice_action.append(Action)
    return action


## Phase d'entrainement

for epsilon in EPSILON:

    for alpha in ALPHA:

        for discount_factor in DISCOUNT_FACTOR:

            Q = np.zeros((state_size,
                          action_size))  ## Q_table du problème, Q[s][a] correspond à la récompense attendue si on applique l'action a à l'état s
            Matrice_état = []  ## matrice contenant les états connus
            Matrice_action = []  ## matrice contenant les actions connues

            REWARD = []
            for i_episode in range(
                    125):  # L'apprentissage d'un modèle consiste en 125 tests sucessifs, on fait 125 parties

                State = env.reset()  # a chaque test on réinitialise l'environnement, ici State correspond à l'état initial

                ## Transformation de l'état en une liste de 9 nombres que l'on va rajouter ensuite dans la matrice d'état
                ## Autrement dit on représente l'état de notre palette par une liste de 9 éléments qui correspond aux lignes du dictionnaire mise bout à bout
                State = State['fill']

                state = []
                state.extend(State[0])
                state.extend(State[1])
                state.extend(State[2])

                total_reward = 0

                done = False
                t = 0

                while not done:  ## on entre dans la partie
                    t += 1
                    env.render()

                    # rajoute l'état appris à la matrice d'état si on ne le connait pas

                    if state not in Matrice_état:
                        Matrice_état.append(state)

                    # Indice de l'état dans la matrice état
                    index_state = Matrice_état.index(state)
                    # print('state : ',state)

                    # Choix de l'action
                    action = action_choice(Q, Matrice_état, Matrice_action, epsilon, state)

                    # Indice de l'action dans la matrice d'action
                    Action = Converter(action)
                    index_action = Matrice_action.index(Action)

                    # Transition à l'état suivant
                    Next_state, reward, done, _ = env.step(
                        action)  # reward correspond à +1 à chaque tour qui passe c'est à dire à chaque fois qu'on rajoute une caisse, peut on la modifier pour que ce soit plus précis et inclure les actions à ne vraiment pas prendre comme mettre une caisse en dehors

                    total_reward = total_reward + reward

                    # transformation de l'état en liste que l'on rajoute à la matrice d'état

                    Next_state = Next_state['fill']
                    next_state = []
                    next_state.extend(Next_state[0])
                    next_state.extend(Next_state[1])
                    next_state.extend(Next_state[2])

                    if next_state not in Matrice_état:
                        Matrice_état.append(next_state)

                    index_next_state = Matrice_état.index(next_state)

                    # corrige Q_table
                    best_next_action = np.argmax(Q[
                                                     index_next_state])  # choisit la meilleure action pour l'état suivant, retourne l'indice de la meilleure action
                    td_target = reward + discount_factor * Q[index_next_state][best_next_action]  # max(Q(s',a'))
                    td_delta = td_target - Q[index_state][
                        index_action]  # Q(s,a), ici on fait la différence entre la valeur "optimale" et Q(s,a), cette différence va tendre vers 0 au fur et à mesure des itérations
                    Q[index_state][index_action] += alpha * td_delta

                    state = next_state

                    if done:
                        env.render()
                        REWARD = REWARD + [total_reward]
                        print("Episode" + str(i_episode) + "finished after {} timesteps".format(t + 1))
                        break

                epsilon = MIN_EPSILON + (epsilon - MIN_EPSILON) * np.exp(
                    -DECAY_RATE * i_episode)  # réduction d'espilon au fur et à mesure des parties afin de réduire de plus en plus l'exploration
            print(alpha, discount_factor, epsilon, maxListe(REWARD))
            env.close()