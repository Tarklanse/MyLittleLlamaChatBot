from langchain_core.tools import tool
from typing import Annotated, List
from .weaviateVectorStoreHandler import viewVector
import random,os
from django.conf import settings

BASE_Files_PATH = "ai_txt"

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

@tool
def genRandomNumber(a: int, b: int) -> int:
    """This function will generate a Random number in Range"""
    if a>b:
        return random.randint(b,a)
    else:
        return random.randint(a,b)

@tool
def findBigger(a: int,b: int) -> int:
    """This function will compare which number is bigger,and return the biggest one."""
    if a>b:
        return a
    else:
        return b

@tool
def sortList_asc(input_a:List[int]) -> List[int]:
    """This function will helping sorting numbers in asc."""
    return sorted(input_a, reverse=True)
    
@tool
def sortList_desc(input_a:List[int]) -> List[int]:
    """This function will helping sorting numbers in desc."""
    return sorted(input_a)
    

@tool
def is_prime(input_a: int) -> bool:
    """This function will check a number is prime or not"""
    if input_a == 2 or input_a == 3: return True
    if input_a < 2 or input_a%2 == 0: return False
    if input_a < 9: return True
    if input_a%3 == 0: return False
    r = int(input_a**0.5)
    f = 5
    while f <= r:
        #print('\t',f)
        if input_a % f == 0: return False
        if input_a % (f+2) == 0: return False
        f += 6
    return True

@tool
def find_factors(input_a:int) -> List[int]:
    """This function will find a number's factors and return as a list"""
    if input_a <= 0:
        raise ValueError("The number must be a positive integer.")
    factors = []
    for i in range(1, int(input_a ** 0.5) + 1):
        if input_a % i == 0:
            factors.append(i)
            if i != input_a // i:  # Avoid duplicates for perfect squares
                factors.append(input_a // i)
    return sorted(factors)

@tool
def query_vector(chat_id:str,message:str) -> str:
    """This function can let you query current chat's vector store"""
    return viewVector(chat_id,message)

@tool
def write_file(input_filename: str, input_context: str) -> str:
    """
    This function allows you to write or edit a text file, and it returns the file path.

    Parameters:
    - input_filename (str): The name of the file to write to.
    - input_context (str): The content to write to the file.

    Returns:
    - str: The path can download the written file,put this in response so user can download the file.
    """
    target_dir = os.path.join(settings.AI_TEXT_PATH)
    os.makedirs(target_dir, exist_ok=True)
    
    filepath=os.path.join(target_dir,input_filename)
    with open(filepath, 'w', encoding='utf-8') as file:
        file.write(input_context)
    # Return the relative URL
    relative_url = f"/api/download_AI_Gen_file/{input_filename}"
    return relative_url
