# profile_one.py   usage: python profile_one.py <variant> <S>
def main(variant: str, S: int, block: int = 256) -> None:
    # 1 build input  2 call 3 times to warm up  3 torch.cuda.synchronize()
    # 4 torch.cuda.cudart().cudaProfilerStart()  5 one call  6 synchronize
    # 7 torch.cuda.cudart().cudaProfilerStop()
    ...