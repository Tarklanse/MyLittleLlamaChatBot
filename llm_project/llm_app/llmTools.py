from langchain_core.tools import tool
from typing import Annotated, List

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

@tool 
def findBigger(a:int,b:int) -> int:
    """This function will compare which number is bigger,and return the biggest one."""
    if a>b:
        return a
    else:
        return b

@tool
def sortList(input_a:[List[int],"the list want to sort."],input_b:[bool,"the way to sort list,True for from low to high,False for from high to low."]):
    """This function will helping sorting numbers."""
    if input_b:
        return sorted(input_a)
    else:
        return sorted(input_a, reverse=True)
