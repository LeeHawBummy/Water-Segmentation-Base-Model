import torch
import numpy as np

def custom_collate(batch):
    """
    自定义collate函数，解决tensor storage resizing的问题
    这个版本直接处理tensor拼接，避免使用default_collate
    """
    # 过滤掉None值
    batch = [item for item in batch if item is not None]
    
    if len(batch) == 0:
        return None
    
    # 手动处理每个key的tensor拼接
    collated_batch = {}
    
    for key in batch[0].keys():
        if key == 'case_name':
            # 处理字符串列表
            collated_batch[key] = [item[key] for item in batch]
        else:
            # 处理tensor
            tensors = [item[key] for item in batch]
            
            # 确保所有tensor都是torch.Tensor
            processed_tensors = []
            for tensor in tensors:
                if not isinstance(tensor, torch.Tensor):
                    tensor = torch.from_numpy(tensor.copy())
                else:
                    # 确保tensor是连续的
                    tensor = tensor.contiguous()
                processed_tensors.append(tensor)
            
            # 手动拼接tensor
            try:
                collated_batch[key] = torch.stack(processed_tensors, dim=0)
            except RuntimeError as e:
                # 如果stack失败，尝试使用cat
                if "must have the same shape" in str(e):
                    # 对于不同形状的tensor，使用padding
                    max_shape = [max(s[i] for s in [t.shape for t in processed_tensors]) for i in range(len(processed_tensors[0].shape))]
                    padded_tensors = []
                    for tensor in processed_tensors:
                        # 计算padding
                        pad_width = [(0, max_shape[i] - tensor.shape[i]) for i in range(len(tensor.shape))]
                        # 只在需要的地方添加padding
                        if any(p[1] > 0 for p in pad_width):
                            padded_tensor = torch.nn.functional.pad(tensor, pad_width[::-1])
                        else:
                            padded_tensor = tensor
                        padded_tensors.append(padded_tensor)
                    collated_batch[key] = torch.stack(padded_tensors, dim=0)
                else:
                    raise e
    
    return collated_batch