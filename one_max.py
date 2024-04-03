"""
OneMax問題を解く。
OneMax問題：[0, 1, 1, 1, 0, 1, 0, 1, 1, 0]のように0/1からなる数列の和を最大化する問題.
和が最大となるには, すべての要素が1となることを目指す.
"""

import deap
from deap import base
from deap import creator
from deap import tools
from deap import algorithms

import random
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 問題特有の定数の定義
ONE_MAX_LENGTH = 100 # 最適化対象の配列の長さ

# GAに関する定数の定義
POPULATION_SIZE = 200 # 人口
P_CROSSOVER = 0.9 # 交叉を発生させる確率
P_MUTATION = 0.1 # 突然変異を発生させる確率
MAX_GENERATIONS = 50 # 最大世代数
HALL_OF_FAME_SIZE = 10 #

# 実行するたびに振る舞いが同じになるように乱数のシード値を設定しておく
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# ユーティリティ機能を提供する「ToolBox」の作成
toolbox = base.Toolbox()

# 0か1をランダムに返す関数を登録する。
toolbox.register("zeroOrOne", random.randint, 0, 1)

# 戦略：適応度関数を最大化する。
creator.create("FitnessMax", base.Fitness, weights=(1.0,))

# 個体を定義する。listを基底クラスとする。listは遺伝子配列を表す。
creator.create("Individual", list, fitness=creator.FitnessMax)

# 個体を生成する処理を定義する。 0/1で長さ100の配列を生成する。
toolbox.register("individualCreator", tools.initRepeat, creator.Individual, toolbox.zeroOrOne, ONE_MAX_LENGTH)

# 集団（個体群）を生成する処理を定義する。
toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)

# 適合度関数の設定
def oneMaxFitness(individual):
    """
    :param individual: Individualクラスのインスタンス。listを継承している。
    :return: 適合度。今回は各個体の遺伝子配列の和。
    """
    return sum(individual), # tupleを返す必要がある

toolbox.register("evaluate", oneMaxFitness)

#-------------------------------------------------------------------
# 世代更新方法の設定
#-------------------------------------------------------------------

# トーナメント選択のサイズを増やしすぎると弱い個体が生き残る確率が減っていき、多様性が失われる（早期収束と呼ばれる現象が起きる）。
# OneMax問題はそこまで複雑ではないため、トーナメントサイズを大きくしても突然変異によって多様性の確保ができている。
# 突然変異の確率を小さくすると最適解にたどり着くのが非常に遅くなる。
toolbox.register("select", tools.selTournament, tournsize=3)
# toolbox.register("select", tools.selRoulette)

toolbox.register("mate", tools.cxOnePoint) # 一点交叉
# toolbox.register("mate", tools.cxTwoPoint) # 二点交叉

toolbox.register("mutate", tools.mutFlipBit, indpb=1.0/ONE_MAX_LENGTH) # 突然変異：ビット反転

def main_long():
    population = toolbox.populationCreator(n=POPULATION_SIZE)
    generationCounter = 0

    # population内のすべての個体に対して、evaluate関数を適用する。
    fitnessValues = list(map(toolbox.evaluate, population))

    for individual, fitnessValue in zip(population, fitnessValues):
        individual.fitness.values = fitnessValue # fitnessValueはTupleのはず。

    fitnessValues = [individual.fitness.values[0] for individual in population]

    maxFitnessValues = []
    meanFitnessValues = []

    while max(fitnessValues) < ONE_MAX_LENGTH and generationCounter < MAX_GENERATIONS:
        generationCounter = generationCounter + 1

        # 次の世代を選択（トーナメント選択）
        offspring = toolbox.select(population, len(population))

        # このあとの操作が元の集団に影響を及ぼさないようにCloneしておく。
        offspring = list(map(toolbox.clone, offspring))

        # 交叉：偶数番目と奇数番目の子たちで交叉させる
        # childはIndividualクラスのインスタンス
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < P_CROSSOVER:
                toolbox.mate(child1, child2) # 一点交叉
                del child1.fitness.values
                del child2.fitness.values

        # 突然変異：ビット反転
        # mutantはIndividualクラスのインスタンス
        for mutant in offspring:
            if random.random() < P_MUTATION:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # 適合度をまだ計算していない個体を集める
        freshIndividuals = [ind for ind in offspring if not ind.fitness.valid]

        # 集めた個体の適合度を計算。List[Tuple] のはず。
        freshFitnessValues = list(map(toolbox.evaluate, freshIndividuals))

        for individual, fitnessValue in zip(freshIndividuals, freshFitnessValues):
            individual.fitness.values = fitnessValue # Tupleのはず。

        # 現在の子孫で集団を置き換える
        population[:] = offspring

        # List[Tuple[float,]] -> List[float]
        fitnessValues = [ind.fitness.values[0] for ind in population]

        maxFitness = max(fitnessValues)
        meanFitness = sum(fitnessValues) / len(population)
        maxFitnessValues.append(maxFitness)
        meanFitnessValues.append(meanFitness)
        print(f"- Generation {generationCounter}: Max Fitness = {maxFitness}, Avg Fitness = {meanFitness}")

        best_index = fitnessValues.index(maxFitness)
        print("Best Individual = ", *population[best_index], "\n")

    sns.set_style("whitegrid")
    plt.plot(maxFitnessValues, color="red", label="Max Fitness")
    plt.plot(meanFitnessValues, color="green", label="Avg Fitness")
    plt.xlabel("Generation")
    plt.ylabel("Max / Avg Fitness")
    plt.title("Max and Average Fitness over Generations")
    plt.legend()
    plt.show()
    return


def main_short():
    """
    main_long()に対し、StatisticsとLogbookを用いることでソースコードを短縮した関数。
    """

    population = toolbox.populationCreator(n=POPULATION_SIZE)

    # 統計情報を記録するオブジェクトの生成
    stats = tools.Statistics(lambda ind : ind.fitness.values)
    stats.register("max", np.max)
    stats.register("avg", np.mean)

    # 遺伝的アルゴリズムの一連のフローを実行する。
    population, logbook = algorithms.eaSimple(population, toolbox, cxpb=P_CROSSOVER, mutpb=P_MUTATION, ngen=MAX_GENERATIONS,
                                              stats=stats, verbose=True)

    # 記録されている統計情報を取得。
    maxFitnessValues, meanFitnessValues = logbook.select("max", "avg")

    # 可視化
    sns.set_style("whitegrid")
    plt.plot(maxFitnessValues, color="red", label="Max Fitness")
    plt.plot(meanFitnessValues, color="green", label="Avg Fitness")
    plt.xlabel("Generation")
    plt.ylabel("Max / Avg Fitness")
    plt.title("Max and Average Fitness over Generations")
    plt.legend()
    plt.show()

    return


def main_short_hof():
    """
    short()に対して Hall of Fame (hof, 殿堂入り) を利用できる。
    殿堂入り：「選択」「交叉」「突然変異」などで良い個体が失われてしまう際、その個体を保持しておく機能。
    """

    population = toolbox.populationCreator(n=POPULATION_SIZE)

    stats = tools.Statistics(lambda ind : ind.fitness.values)
    stats.register("max", np.max)
    stats.register("avg", np.mean)

    # hofオブジェクトを生成
    hof = tools.HallOfFame(HALL_OF_FAME_SIZE)

    # 遺伝的アルゴリズム 全処理フローを実行
    population, logbook = algorithms.eaSimple(population, toolbox, cxpb=P_CROSSOVER, mutpb=P_MUTATION,
                                              ngen=MAX_GENERATIONS, stats=stats, halloffame=hof, verbose=True)

    # hofのアイテムを表示。
    print("Hall of Fame Individuals = ", *hof.items, sep="\n")
    print("Best Ever Individual = ", hof.items[0])

    # 統計情報を取得。
    maxFitnessValues, meanFitnessValues = logbook.select("max", "avg")

    sns.set_style("whitegrid")
    plt.plot(maxFitnessValues, color="red", label="Max Fitness")
    plt.plot(meanFitnessValues, color="green", label="Avg Fitness")
    plt.xlabel("Generation")
    plt.ylabel("Max / Avg Fitness")
    plt.title("Max and Average Fitness over Generations")
    plt.legend()
    plt.show()
    return

if __name__ == "__main__":
    # main_long()
    # main_short()
    main_short_hof()