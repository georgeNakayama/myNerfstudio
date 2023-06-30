import torch 
import torch.nn as nn
from typing import Callable, List, Literal, Any
from nerfstudio.cameras.rays import RaySamples
from itertools import combinations 
from chamferdist import ChamferDistance

class BasePairWiseDistance(nn.Module):
    def __init__(self, eval_func: Callable, reduction: Literal["sum", "mean", "none"]="mean"):
        super().__init__()
        self.eval_func = eval_func
        self.reduction=reduction
        
    def forward(self, sample_list: List[RaySamples]):
        all_dists = []
        for s1, s2 in combinations(sample_list, 2):
            dist = self.eval_func(s1, s2)
            
            # assert s1.spacing_to_euclidean_fn is not None and s2.spacing_to_euclidean_fn is not None
            # s1_starts = s1.spacing_to_euclidean_fn(s1.spacing_starts)
            # s1_ends = s1.spacing_to_euclidean_fn(s1.spacing_ends)
            # s2_starts = s2.spacing_to_euclidean_fn(s2.spacing_starts)
            # s2_ends = s2.spacing_to_euclidean_fn(s2.spacing_ends)
            # s1_medians = (s1_starts + s1_ends) / 2.0
            # s2_medians = (s2_starts + s2_ends) / 2.0
            # dist = self.eval_func(s1_medians, s2_medians)
            all_dists.append(dist)
        all_dists = torch.stack(all_dists, dim=0)
        if self.reduction == "sum":
            all_dists = all_dists.sum(0)
        elif self.reduction == "mean":
            all_dists = all_dists.mean(0)
        return all_dists
    
class ChamferPairWiseDistance(BasePairWiseDistance):
    def __init__(self, reduction: Literal['sum', 'mean', 'none'] = "mean"):
        chamfer = ChamferDistance()
        def eval_func(x, y):
            return chamfer(x, y, bidirectional=True, reduction=None)
            
        super().__init__(eval_func, reduction)
        
if __name__ == "__main__":
    a = torch.randn([400*800, 256, 1]).cuda()
    b = torch.randn([400*800, 256, 1]).cuda()
    fn = ChamferPairWiseDistance()
    print(fn([a, b]))
    

        