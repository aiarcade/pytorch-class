import torch
import torch.multiprocessing as mp

def mul_function(rank, world_size, tensor, scalar):

    start = rank * len(tensor) // world_size
    end = (rank + 1) * len(tensor) // world_size
    sub_tensor = tensor[start:end]
    result = sub_tensor * scalar
    print("Rank",rank,result)


if __name__ == "__main__":
    tensor = torch.arange(1, 101)
    world_size = torch.cuda.device_count()
    scalar = 2
    

    mp.spawn(mul_function, args=(world_size, tensor, scalar), nprocs=world_size)

    results = []

