import math
import torch


def cluster_groups(recompute_ptrs: list[tuple[int, int]]) -> list[int]:
    new_recompute_ptrs = []
    cur_group = recompute_ptrs[0]
    for start, end in recompute_ptrs[1:]:
        if start > cur_group[1]:
            new_recompute_ptrs.append(cur_group)
            cur_group = [start, end]
            continue
        cur_group = [min(start, cur_group[0]), max(cur_group[1], end)]
    return new_recompute_ptrs + [cur_group]

def exp_function(x, r=0.2):
    return math.exp(r * x)

def log_function(x, r=0.2):
    return math.log(r * x)

def linear_function(x, r=1):
    return r * x

class DARICPolicy:
    def __init__(
        self, 
        avg_num: int = 32,
        r: float = 0.1,
        function: str = "linear",
        
    ):
        self.alpha = 1
        self.r = r
        self.avg_num = avg_num
        if function == "linear":
            self.function = linear_function
        elif function == "exp":
            self.function = exp_function
        elif function == "log":
            self.function = log_function
        else:
            raise ValueError(f"Unknown function: {function}")

    def get_recompute_ptrs(self, group_ptr: list[int]) -> list[int]:
        num_groups = len(group_ptr) - 1
        budget = self.avg_num * num_groups
        if num_groups == 1:
            return []
        
        if num_groups == 2:
            n = int(budget / 2)
            recompute_ptrs = [
                [group_ptr[1], min(group_ptr[1] + n, group_ptr[2])],
                [max(0, group_ptr[2] - n), group_ptr[2] - 1],
            ]
            return cluster_groups(recompute_ptrs)

        f2 = self.function(1, self.alpha)
        fn = self.function(num_groups-1, self.alpha)
        f_sum = sum([self.function(i, self.alpha) for i in range(1, num_groups)])
        
        A = torch.tensor([
            [fn, 1, -1],
            [f2, 1,  0], 
            [f_sum, num_groups - 1, 1]
        ]).float()

        c = torch.tensor([0, self.r * self.avg_num, budget], dtype=torch.float32)
        params = torch.linalg.solve(A, c)
        
        a, b, n_l = params.tolist()
        recompute_ptrs = []
        for i in range(1, num_groups):
            num = a * self.function(i, self.alpha) + b
            num = max(1, round(num))
            start = group_ptr[i]
            end = min(group_ptr[i] + num, group_ptr[i + 1])
            recompute_ptrs.append([start, end])

        if recompute_ptrs[-1][1] == group_ptr[-1]:
            recompute_ptrs[-1][1] -= 1
        else:
            last_num = round(n_l)
            if last_num > 1:
                recompute_ptrs.append(
                    [max(0, group_ptr[-1] - 1 - last_num), group_ptr[-1] - 1]
                )
        recompute_ptrs = cluster_groups(recompute_ptrs)
        remain_budget = budget - sum(
            [end - start for start, end in recompute_ptrs]
        )
        i = 0
        while i < 2:
            recompute_ptrs[-1][0] = max(0, recompute_ptrs[-1][0] - remain_budget)
            recompute_ptrs = cluster_groups(recompute_ptrs)
            remain_budget = budget - sum(
                [end - start for start, end in recompute_ptrs]
            )
            if remain_budget == 0:
                break
            i += 1
        return recompute_ptrs
