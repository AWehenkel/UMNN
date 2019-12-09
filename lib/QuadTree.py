import numpy as np
import torch

class QuadCell():
    def __init__(self, x_min, x_max, y_min, y_max, value):
        self.x_max = x_max
        self.x_min = x_min
        self.y_max = y_max
        self.y_min = y_min
        self.qc = None
        self.value = value
        self.new_leaf = True

    def update_value(self):
        if self.qc is None:
            return self.value
        val = 0.
        for qc in self.qc:
            val += qc.update_value()
        self.value = val
        return val

    def set_value(self, value):
        self.value = value
        self.new_leaf = False

    def center(self):
        return self.x_min + (self.x_max - self.x_min)/2., self.y_min + (self.y_max - self.y_min)/2.

    def area(self):
        return (self.x_max - self.x_min) * (self.y_max - self.y_min)

    def flattened_leafs(self):
        if self.qc is None:
            centers_x = [self.x_min + (self.x_max - self.x_min) / 2.]
            centers_y = [self.y_min + (self.y_max - self.y_min) / 2.]
            values = [self.value]
            areas = [self.area()]
        else:
            centers_x = []
            centers_y = []
            values = []
            areas = []
            for qc in self.qc:
                x, y, v, a = qc.flattened_leafs()
                centers_x += x
                centers_y += y
                values += v
                areas += a
        return centers_x, centers_y, values, areas

    def flattened_new_leafs(self):
        if self.qc is None:
            if self.new_leaf:
                centers_x = [self.x_min + (self.x_max - self.x_min) / 2.]
                centers_y = [self.y_min + (self.y_max - self.y_min) / 2.]
                areas = [self.area()]
                leafs = [self]
            else:
                centers_x = []
                centers_y = []
                areas = []
                leafs = []
        else:
            centers_x = []
            centers_y = []
            areas = []
            leafs = []
            for qc in self.qc:
                x, y, a, l = qc.flattened_new_leafs()
                centers_x += x
                centers_y += y
                areas += a
                leafs += l
        return centers_x, centers_y, areas, leafs

    def sample_children(self, exploration_ratio=.4):
        if self.qc is None:
            self.qc = []
            self.new_leaf = False
            delta_x = self.x_max - self.x_min
            delta_y = self.y_max - self.y_min
            self.qc.append(QuadCell(self.x_min, self.x_max - delta_x/2, self.y_min, self.y_max - delta_y/2, self.value/4))
            self.qc.append(QuadCell(self.x_min + delta_x/2, self.x_max, self.y_min, self.y_max - delta_y/2, self.value/4))
            self.qc.append(QuadCell(self.x_min, self.x_max - delta_x/2, self.y_min + delta_y/2, self.y_max, self.value/4))
            self.qc.append(QuadCell(self.x_min + delta_x/2, self.x_max, self.y_min + delta_y/2, self.y_max, self.value/4))
            return self.qc

        r = np.random.random() * self.value
        explore = 1 if np.random.random() < exploration_ratio else 0
        counter = 0.
        for qc in self.qc:
            counter += explore * self.value/4. + (1 - explore) * qc.value
            if r < counter:
                return qc.sample_children()
        return None


def quad_tree_sampling(x_range, y_range, n_parallel_split, n_iteration, density):
    root = QuadCell(x_range[0], x_range[1], y_range[0], y_range[1], 1.)
    for j in range(n_iteration):
        for i in range(n_parallel_split):
            root.sample_children()
        x, y, areas, leafs = root.flattened_new_leafs()
        x = torch.tensor(x)
        y = torch.tensor(y)
        areas = torch.tensor(areas)
        new_values = density(x, y) * areas
        for leaf_id in range(len(leafs)):
            leafs[leaf_id].set_value(new_values[leaf_id].item())
        root.update_value()

    return root.flattened_leafs(), root


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import math
    root = QuadCell(-5, 5, -5, 5, 1.)

    for i in range(1000):
        print(i)
        cells = root.sample_children()
        for c in cells:
            c_center = c.center()
            val = c.area() * 1/(2*math.pi)*np.exp(-.5*((c_center[0] - 1.)**2 + c_center[1]**2))
            root.update_value()
            c.set_value(val)

    x, y, v, _ = root.flattened_leafs()
    print(root.update_value(), np.array(v).sum())
    plt.scatter(x, y, alpha=.1)
    plt.show()