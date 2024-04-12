"""
ナップサック問題(0-1問題)に必要なデータを定義する。
0-1問題：選択する・しないの2値を決める問題。1つのアイテムを複数選択はできない。
"""

import numpy as np

class Knapsack01Problem:
    def __init__(self):
        # initialize instance variables:
        self.items: list[tuple] = []
        self.max_capacity: int = 0

        # initialize the data:
        self.__init_data()

    def __len__(self):
        return len(self.items)

    def __init_data(self):
        """RosettaCode.orgというサイトのサンプルデータを生成する。"""
        self.items = [
            ("map", 9, 150), # (名称, 重さ, 価値)
            ("compass", 13, 35),
            ("water", 153, 200),
            ("sandwich", 50, 160),
            ("glucose", 15, 60),
            ("tin", 68, 45),
            ("banana", 27, 60),
            ("apple", 39, 40),
            ("cheese", 23, 30),
            ("beer", 52, 10),
            ("suntan cream", 11, 70),
            ("camera", 32, 30),
            ("t-shirt", 24, 15),
            ("trousers", 48, 10),
            ("umbrella", 73, 40),
            ("waterproof trousers", 42, 70),
            ("waterproof overclothes", 43, 75),
            ("note-case", 22, 80),
            ("sunglasses", 7, 20),
            ("towel", 18, 12),
            ("socks", 4, 50),
            ("book", 30, 10)
        ]

        self.max_capacity = 400

    def get_value(self, zero_one_list: list[int]):
        """
        :param zero_one_list: 0か1が格納されたリスト。 1ならそのアイテムは選択状態。
        :return: 選択したアイテムの価値の総量。
        """
        iter_max = min(len(zero_one_list), self.__len__())
        total_weight = 0
        total_value = 0

        for i in range(iter_max):
            item, weight, value = self.items[i]
            if total_weight + weight <= self.max_capacity:
                total_weight += zero_one_list[i] * weight
                total_value += zero_one_list[i] * value

        return total_value

    def print_items(self, zero_one_list: list[int]):
        """
        現在の選択状態をprintする。
        :param zero_one_list:
        """
        iter_max = min(len(zero_one_list), self.__len__())
        total_weight = 0
        total_value = 0

        for i in range(iter_max):
            item, weight, value = self.items[i]
            if total_weight + weight <= self.max_capacity:
                total_weight += zero_one_list[i] * weight
                total_value += zero_one_list[i] * value
            if zero_one_list[i] > 0:
                print(f"- Adding {item}: weight = {weight}, value = {value}, accumulated weight = {total_weight}, accumulated value = {total_value}")

        print(f"- Total weight = {total_weight}, Total value = {total_value}")

def main():
    knapsack = Knapsack01Problem()

    random_solution = np.random.randint(2, size=len(knapsack))
    print("Random Solution = ", random_solution)
    knapsack.print_items(random_solution)

if __name__ == "__main__":
    main()