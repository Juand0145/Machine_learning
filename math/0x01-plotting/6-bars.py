#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))

people = ("Farrah", "Fred", "Felicia")
fruits = ("apples", "bananas", "oranges", "peaches")


colors = ["red", "yellow", "#ff8000", "#ffe5b4"]
n_rows = len(fruit)

bar_width = 0.5


y_offset = np.zeros(len(people))



for row in range(n_rows):
    plt.bar(people, fruit[row], bar_width, bottom=y_offset,
            color=colors[row])
    y_offset = y_offset + fruit[row]

plt.ylabel("Quantity of Fruit")
plt.title("Number of Fruit per Person")
plt.ylim([0, 80])


plt.legend(labels=fruits)
plt.show()
