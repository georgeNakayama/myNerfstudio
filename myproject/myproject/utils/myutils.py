import torch 
from torch import Tensor 
from typing import Union, Dict, List, Any, Tuple, Mapping
from collections import defaultdict
from nerfstudio.cameras.rays import RaySamples

def sync_data(key: str, data: Union[Tensor, List[Tensor], Dict[str, Tensor]], output_list: Dict[str, Any]) -> Dict[str, Union[List[Any], List[List[Any]], Dict[str, List[Any]]]]:
    if torch.is_tensor(data):
        if key not in output_list:
            output_list[key] = []
        output_list[key].append(data.detach().cpu())
    elif isinstance(data, list):
        if key not in output_list:
            output_list[key] = [[] for _ in len(data)]
        for i, datum in enumerate(data):
            assert torch.is_tensor(datum)
            output_list[key][i].append(datum.detach().cpu())
    elif isinstance(data, dict):
        if key not in output_list:
            output_list[key] = defaultdict(list)
        for k, v in data.items():
            assert torch.is_tensor(v) or RaySamples
            output_list[key][k].append(v.to("cpu"))
    else:
        raise NotImplementedError
    return output_list
