import os
import subprocess
import numpy as np

def main():
    print("Testing evaluate.py robustness...")
    os.makedirs("test_input", exist_ok=True)
    os.makedirs("test_output", exist_ok=True)
    
    # Create non-square, non-multiple-of-16 arrays
    shapes = [(250, 250), (257, 301), (15, 17)]
    for shape in shapes:
        arr = np.random.rand(*shape).astype(np.float32)
        np.save(f"test_input/test_{shape[0]}x{shape[1]}.npy", arr)
        
    print(f"Generated {len(shapes)} tricky shapes. Running evaluate.py...")
    
    # Run evaluate
    result = subprocess.run(["python", "evaluate.py", "-i", "test_input", "-o", "test_output"], capture_output=True, text=True)
    
    if result.returncode != 0:
        print("FAILED!")
        print(result.stderr)
    else:
        print("SUCCESS! evaluate.py ran without crashing.")
        print(result.stdout)
        
        # Verify output shapes
        for shape in shapes:
            out_arr = np.load(f"test_output/test_{shape[0]}x{shape[1]}.npy")
            expected_shape = (shape[0]*2, shape[1]*2)
            if out_arr.shape == expected_shape:
                print(f"  Shape {shape} correctly upsampled to {expected_shape}")
            else:
                print(f"  ERROR: Shape {shape} resulted in {out_arr.shape} instead of {expected_shape}")

if __name__ == "__main__":
    main()
