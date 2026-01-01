#!/usr/bin/env python3
"""
Comprehensive test file demonstrating all fixed features.
Tests classes, async, context managers, pattern matching, and more.
"""

import asyncio
from typing import Optional
from dataclasses import dataclass


# Test 1: Class definitions (previously broken)
@dataclass
class User:
    """User data class"""
    name: str
    age: int
    
    def validate(self) -> bool:
        """Validate user data"""
        if self.age < 0:
            raise ValueError("Age cannot be negative")
        return True


# Test 2: Async functions (previously not distinguished)
async def fetch_data(url: str) -> dict:
    """Async data fetching"""
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.json()


# Test 3: Context managers (previously fell through)
def read_config(filepath: str) -> dict:
    """Context manager usage"""
    config = {}
    
    with open(filepath, 'r') as f:
        for line in f:
            key, value = line.strip().split('=')
            config[key] = value
    
    return config


# Test 4: Pattern matching (Python 3.10+, previously unsupported)
def handle_command(command: str) -> str:
    """Pattern matching example"""
    match command.split():
        case ["quit"] | ["exit"]:
            return "Goodbye!"
        case ["echo", *words]:
            return " ".join(words)
        case ["calc", a, "+", b]:
            return str(int(a) + int(b))
        case _:
            return "Unknown command"


# Test 5: Complex branching (tests branch split fix)
def complex_logic(x: int, y: int, z: int) -> str:
    """Complex control flow with proper branching"""
    if x > 0:
        if y > 0:
            result = "both positive"
        else:
            result = "x positive, y negative"
    else:
        if z > 0:
            result = "x negative, z positive"
        else:
            result = "x and z negative"
    
    return result


# Test 6: Exception handling with finally
def safe_division(a: int, b: int) -> Optional[float]:
    """Exception handling with finally block"""
    result = None
    
    try:
        result = a / b
    except ZeroDivisionError:
        print("Cannot divide by zero")
    except TypeError:
        print("Invalid types")
    finally:
        print("Cleanup executed")
    
    return result


# Test 7: Generators and yield
def fibonacci(n: int):
    """Generator function with yield"""
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b


# Test 8: Multiple return paths (tests terminal nodes)
def categorize(value: int) -> str:
    """Multiple return statements"""
    if value < 0:
        return "negative"
    elif value == 0:
        return "zero"
    elif value < 100:
        return "small"
    else:
        return "large"


# Test 9: Assertions
def validate_positive(x: int) -> int:
    """Assert statement handling"""
    assert x > 0, "Value must be positive"
    return x * 2


# Test 10: Global and nonlocal
counter = 0

def increment():
    """Global variable modification"""
    global counter
    counter += 1
    return counter


# Test 11: Augmented assignments
def process_data(data: list) -> list:
    """Various assignment types"""
    result: list = []  # Annotated assignment
    count = 0          # Regular assignment
    
    for item in data:
        count += 1     # Augmented assignment
        result.append(item * 2)
    
    return result


# Test 12: Unicode in code (tests unicode rendering fix)
def greet_world():
    """Unicode string handling"""
    emoji = "🚀 Hello 世界! ★"
    return emoji


# Test 13: Nested loops (tests depth tracking)
def matrix_multiply(a: list, b: list) -> list:
    """Nested loop structure"""
    result = []
    
    for i in range(len(a)):
        row = []
        for j in range(len(b[0])):
            sum_val = 0
            for k in range(len(b)):
                sum_val += a[i][k] * b[k][j]
            row.append(sum_val)
        result.append(row)
    
    return result


# Test 14: Break and continue statements
def find_first_even(numbers: list) -> Optional[int]:
    """Break and continue usage"""
    for num in numbers:
        if num < 0:
            continue  # Skip negative numbers
        if num % 2 == 0:
            return num  # Found first even
            break       # Unreachable but tests break
    
    return None


# Test 15: Try with multiple handlers
def parse_input(value: str) -> int:
    """Multiple exception handlers"""
    try:
        result = int(value)
    except ValueError as e:
        print(f"Invalid number: {e}")
        result = 0
    except Exception as e:
        print(f"Unexpected error: {e}")
        result = -1
    
    return result


# Test 16: Raise statements
def validate_age(age: int) -> None:
    """Raise statement handling"""
    if age < 0:
        raise ValueError("Age cannot be negative")
    if age > 150:
        raise ValueError("Age unrealistic")


# Test 17: Delete statement
def cleanup_cache(cache: dict, key: str) -> None:
    """Delete statement"""
    if key in cache:
        del cache[key]


# Test 18: Lambda and comprehensions
def functional_operations(data: list) -> dict:
    """Lambdas and comprehensions"""
    # Lambda
    transform = lambda x: x ** 2
    
    # List comprehension
    squares = [transform(x) for x in data if x > 0]
    
    # Dict comprehension
    square_map = {x: x**2 for x in data}
    
    # Set comprehension
    unique_squares = {x**2 for x in data}
    
    return {
        'squares': squares,
        'map': square_map,
        'unique': unique_squares
    }


# Test 19: Walrus operator (assignment expression)
def process_file(filepath: str) -> list:
    """Walrus operator usage"""
    results = []
    
    with open(filepath) as f:
        while (line := f.readline()):
            if (processed := line.strip()):
                results.append(processed)
    
    return results


# Test 20: Complex nested class
class DataProcessor:
    """Complex class with nested logic"""
    
    def __init__(self, data: list):
        self.data = data
        self.results = []
    
    def process(self) -> list:
        """Process data with complex logic"""
        for item in self.data:
            try:
                if isinstance(item, int):
                    if item > 0:
                        self.results.append(item * 2)
                    else:
                        self.results.append(item / 2)
                elif isinstance(item, str):
                    self.results.append(item.upper())
                else:
                    raise TypeError(f"Unsupported type: {type(item)}")
            except Exception as e:
                print(f"Error processing {item}: {e}")
                continue
        
        return self.results
    
    @staticmethod
    def validate(value):
        """Static method"""
        return value is not None
    
    @classmethod
    def from_file(cls, filepath: str):
        """Class method"""
        with open(filepath) as f:
            data = f.readlines()
        return cls(data)


# Entry point
def main():
    """Main function demonstrating all features"""
    print("Testing all features...")
    
    # Test basic operations
    result = complex_logic(1, -1, 1)
    print(f"Result: {result}")
    
    # Test unicode
    greeting = greet_world()
    print(greeting)
    
    # Test class
    processor = DataProcessor([1, 2, "hello", -5])
    processed = processor.process()
    print(f"Processed: {processed}")


if __name__ == "__main__":
    main()