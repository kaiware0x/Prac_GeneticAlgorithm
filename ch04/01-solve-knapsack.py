
from deap import base
from deap import tools
from deap import creator
from deap import algorithms

import random
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import knapsack

knapsack = knapsack.Knapsack01Problem()

def knapsack_value(individual: list[int]) -> tuple:
    return (knapsack.get_value(individual), )

# GAに関する定数の定義
POPULATION_SIZE = 50 # 人口
P_CROSSOVER = 0.9 # 交叉を発生させる確率
P_MUTATION = 0.1 # 突然変異を発生させる確率
MAX_GENERATIONS = 50 # 最大世代数
HALL_OF_FAME_SIZE = 1 # Hall of Fame (hof, 殿堂入り)：「選択」「交叉」「突然変異」などで良い個体が失われてしまう際、その個体を保持しておく機能。

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

"""
手順
1. 適応度関数に対する指標を定義する（最大化or最小化）
2. 個体クラスを定義する
3. 遺伝子生成のための関数を準備する
4. 個体インスタンスの生成処理を定義する
5. 集団の生成処理を定義する（個体インスタンスの集合体）
6. 適応度関数を定義する (evaluate)
7. 遺伝子操作の演算子を定義する (select, mate, mutate)
"""

# 1. 適応度関数に対する指標を定義する（最大化or最小化）
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
# 2. 個体クラスを定義する
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# 3. 遺伝子生成のための関数を準備する
# toolbox.register(関数のエイリアス, 実行する関数, 関数の引数...)
toolbox.register("zeroOrOne", random.randint, 0, 1)
# 4. 個体インスタンスの生成処理を定義する
toolbox.register("individualCreator", tools.initRepeat, creator.Individual, toolbox.zeroOrOne, len(knapsack))

# tools.initRepeat()について：
# tools.initRepeat(container, func, n) -> container(func() for _ in range(n))
# つまりn回分func()を実行し、containerに格納した結果を返す。
# register時にnを指定せず、使用する段階でnを動的に指定することも可能。

# 5. 集団の生成処理を定義する（個体インスタンスの集合体）
toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)

# 6. 適応度関数を定義する (evaluate)
# 適応度関数としてknapsack_value関数を使用する。
toolbox.register("evaluate", knapsack_value)

# 7. 遺伝子操作の演算子を定義する (select, mate, mutate)
# バイナリリストの遺伝子に対応可能な演算子の定義。
toolbox.register("select", tools.selTournament, tournsize=3) # 選択
toolbox.register("mate", tools.cxTwoPoint) # 交叉
toolbox.register("mutate", tools.mutFlipBit, indpb=1.0/len(knapsack)) # 突然変異


# 求解
def main():
    """
    1. 第0世代の集団を生成
    2. 統計情報を記録するオブジェクトを生成
    3. GA実行
    4. 結果表示
    """

    # 第0世代の作成
    population = toolbox.populationCreator(n=POPULATION_SIZE) # initRepeat()のnを動的に与えている。

    # 統計情報を記録するオブジェクトの生成
    stats = tools.Statistics(lambda individual : individual.fitness.values)
    stats.register("max", np.max)
    stats.register("avg", np.mean)

    # Hall-of-Fame オブジェクトの生成
    hof = tools.HallOfFame(HALL_OF_FAME_SIZE)

    # GAの実行
    population, logbook = algorithms.eaSimple(population, toolbox, cxpb=P_CROSSOVER, mutpb=P_MUTATION,
                                             ngen=MAX_GENERATIONS, stats=stats, halloffame=hof,verbose=True)

    # 探索した最適解の表示
    best = hof.items[0]
    print("-- Best Ever Individual = ", best)
    print("-- Best Ever Fitness = ", best.fitness.values[0])
    print("-- Knapsack Items = ")
    knapsack.print_items(best)

    # 統計情報の表示
    max_fitness_values, mean_fitness_values = logbook.select("max", "avg")
    sns.set_style("whitegrid")
    plt.plot(max_fitness_values, color="red", label="Max Fitness")
    plt.plot(mean_fitness_values, color="green", label="Avg Fitness")
    plt.xlabel("Generation")
    plt.ylabel("Max / Avg Fitness")
    plt.title("Max and Average Fitness over Generations")
    plt.legend()
    plt.show()

    return


if __name__ == "__main__":
    main()