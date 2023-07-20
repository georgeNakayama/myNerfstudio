import torch 
import torch.nn as nn
from torch import Tensor
from typing import Callable, List, Literal, Any, Optional, Union
from nerfstudio.cameras.rays import RaySamples
from itertools import combinations 
from chamferdist import ChamferDistance

class BasePairWiseDistance(nn.Module):
    def __init__(self, eval_func: Callable, reduction: Optional[List[Literal["sum", "mean", "max"]]]=["mean"]):
        super().__init__()
        self.eval_func = eval_func
        self.reduction=reduction
        
    def forward(self, sample_list: List[RaySamples]) -> Union[Tensor, List[Tensor]]:
        all_dists = []
        for s1, s2 in combinations(sample_list, 2):
            assert s1.spacing_to_euclidean_fn is not None and s2.spacing_to_euclidean_fn is not None
            assert s1.spacing_starts is not None and s1.spacing_ends is not None 
            assert s2.spacing_starts is not None and s2.spacing_ends is not None 
            s1_starts = s1.spacing_to_euclidean_fn(s1.spacing_starts[..., 0])
            s1_ends = s1.spacing_to_euclidean_fn(s1.spacing_ends[..., 0])
            s2_starts = s2.spacing_to_euclidean_fn(s2.spacing_starts[..., 0])
            s2_ends = s2.spacing_to_euclidean_fn(s2.spacing_ends[..., 0])
            s1_medians = (s1_starts + s1_ends) / 2.0
            s2_medians = (s2_starts + s2_ends) / 2.0
            dist = self.eval_func(s1_medians[..., None], s2_medians[..., None])
            all_dists.append(dist)
        all_dists = torch.stack(all_dists, dim=0)
        
        if self.reduction is None:
            return all_dists
        return_list = []
        for red in self.reduction:
            if red == "sum":
                return_list.append(all_dists.sum(0))
            elif red == "mean":
                return_list.append(all_dists.mean(0))
            elif red == "max":
                return_list.append(all_dists.max(0)[0])
        return return_list
    
class ChamferPairWiseDistance(BasePairWiseDistance):
    def __init__(self, reduction: Optional[List[Literal['sum', 'mean', 'max']]] = ["mean"]):
        chamfer = ChamferDistance()
        def eval_func(x, y):
            return chamfer(x, y, bidirectional=True, reduction=None)
            
        super().__init__(eval_func, reduction)
        
if __name__ == "__main__":
    a = torch.randn([400*800, 256, 1]).cuda()
    b = torch.randn([400*800, 256, 1]).cuda()
    fn = ChamferPairWiseDistance()
    print(fn([a, b]))
    

        