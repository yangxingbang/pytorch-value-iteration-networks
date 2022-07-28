import numpy as np
import matplotlib.pyplot as plt


class obstacles:
    """A class for generating obstacles in a domain"""

    def __init__(self,
                 domsize=None,
                 mask=None,
                 size_max=None,
                 dom=None,
                 obs_types=None,
                 num_types=None):
        self.domsize = domsize or []
        # 传进来的是目的地点
        self.mask = mask or []
        self.dom = dom or np.zeros(self.domsize)
        self.obs_types = obs_types or ["circ", "rect"]
        self.num_types = num_types or len(self.obs_types)
        # 传进来的是障碍物的最大尺寸
        self.size_max = size_max or np.max(self.domsize) / 4

    def check_mask(self, dom=None):
        # Ensure goal is in free space
        if dom is not None:
            # np.any是对每一个维度的元素做或运算
            return np.any(dom[self.mask[0], self.mask[1]])
        else:
            return np.any(self.dom[self.mask[0], self.mask[1]])

    def insert_rect(self, x, y, height, width):
        # Insert a rectangular obstacle into map
        im_try = np.copy(self.dom)
        # 从x,y开始，延展height和width个各自，并全部赋为1
        # 1表示障碍物存在的代价
        im_try[x:x + height, y:y + width] = 1
        return im_try

    def add_rand_obs(self, obj_type):
        # Add random (valid) obstacle to map
        if obj_type == "circ":
            print("circ is not yet implemented... sorry")
        elif obj_type == "rect":
            rand_height = int(np.ceil(np.random.rand() * self.size_max))
            rand_width = int(np.ceil(np.random.rand() * self.size_max))
            randx = int(np.ceil(np.random.rand() * (self.domsize[1] - 1)))
            randy = int(np.ceil(np.random.rand() * (self.domsize[1] - 1)))
            im_try = self.insert_rect(randx, randy, rand_height, rand_width)
        if self.check_mask(im_try):
            return False
        else:
            self.dom = im_try
            return True

    def add_n_rand_obs(self, n):
        # Add random (valid) obstacles to map
        count = 0
        for i in range(n):
            obj_type = "rect"
            if self.add_rand_obs(obj_type):
                count += 1
        return count

    def add_border(self):
        # Make full outer border an obstacle
        im_try = np.copy(self.dom)
        # 有障碍物的格子赋1
        im_try[0:self.domsize[0], 0] = 1
        im_try[0, 0:self.domsize[1]] = 1
        im_try[0:self.domsize[0], self.domsize[1] - 1] = 1
        im_try[self.domsize[0] - 1, 0:self.domsize[1]] = 1
        if self.check_mask(im_try):
            return False
        else:
            self.dom = im_try
            return True

    def get_final(self):
        # Process obstacle map for domain
        im = np.copy(self.dom)
        # 把黑白色颠倒过来
        # max表示把输入拉直后取得最大值
        im = np.max(im) - im
        # 如果不止黑白色的话，可以归一化
        im = im / np.max(im)
        return im

    def show(self):
        # Utility function to view obstacle map
        plt.imshow(self.get_final(), cmap='Greys')
        plt.show()

    def _print(self):
        # Utility function to view obstacle map
        #  information
        print("domsize: ", self.domsize)
        print("mask: ", self.mask)
        print("dom: ", self.dom)
        print("obs_types: ", self.obs_types)
        print("num_types: ", self.num_types)
        print("size_max: ", self.size_max)
