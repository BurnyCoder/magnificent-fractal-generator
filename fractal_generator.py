#!/usr/bin/env python3
"""
Fractal Generator Module - Core functionality for generating different types of fractals.
"""
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from abc import ABC, abstractmethod
import time


class FractalGenerator(ABC):
    """Abstract base class for all fractal generators."""
    
    def __init__(self, width, height, x_min, x_max, y_min, y_max, max_iter, color_scheme):
        self.width = width
        self.height = height
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.max_iter = max_iter
        self.color_scheme = color_scheme
        self.start_time = time.time()
        self.timeout = 15  # Reduced to 15 seconds timeout for calculation
        self.check_frequency = 5  # Check timeout every 5 iterations
        
    @abstractmethod
    def calculate(self):
        """Calculate the fractal (implement in subclasses)."""
        pass
    
    def check_timeout(self):
        """Check if calculation has exceeded the timeout"""
        elapsed = time.time() - self.start_time
        if elapsed > self.timeout:
            print(f"Fractal calculation timed out after {elapsed:.2f} seconds")
            raise TimeoutError(f"Fractal calculation exceeded time limit of {self.timeout} seconds (elapsed: {elapsed:.2f}s)")
    
    def generate_image(self):
        """Generate a PIL image from the fractal data."""
        # Calculate the fractal data
        print(f"Starting fractal calculation: {self.width}x{self.height}, {self.max_iter} iterations")
        start = time.time()
        try:
            data = self.calculate()
            elapsed = time.time() - start
            print(f"Fractal calculation completed in {elapsed:.2f} seconds")
        except Exception as e:
            elapsed = time.time() - start
            print(f"Fractal calculation failed after {elapsed:.2f} seconds: {str(e)}")
            raise
        
        print("Applying color mapping...")
        # Normalize the data for visualization (between 0 and 1)
        if data.max() > 0:
            norm_data = data / data.max()
        else:
            norm_data = data
        
        # Get the colormap
        cmap = plt.get_cmap(self.color_scheme)
        
        # Apply colormap to normalized data
        colored_data = cmap(norm_data)
        
        # Convert to 8-bit RGB values
        colored_data = (colored_data[:, :, :3] * 255).astype(np.uint8)
        
        # Create PIL image
        img = Image.fromarray(colored_data)
        
        print("Image creation complete")
        return img


class MandelbrotSet(FractalGenerator):
    """Mandelbrot set fractal generator."""
    
    def calculate(self):
        """Calculate the Mandelbrot set."""
        # Create a grid of complex numbers
        x = np.linspace(self.x_min, self.x_max, self.width)
        y = np.linspace(self.y_min, self.y_max, self.height)
        c = x[:, np.newaxis] + 1j * y[np.newaxis, :]
        
        # Initialize array for iterations
        iterations = np.zeros((self.width, self.height), dtype=np.float32)
        
        # Initialize z
        z = np.zeros_like(c, dtype=np.complex64)
        
        # Create a mask for points that are still being computed
        mask = np.ones_like(iterations, dtype=bool)
        
        # Iteration counter
        iter_count = 0
        
        # Iterate the Mandelbrot function z = z^2 + c
        while iter_count < self.max_iter and np.any(mask):
            # Check for timeout on regular intervals
            if iter_count % self.check_frequency == 0:
                self.check_timeout()
                
            z[mask] = z[mask] * z[mask] + c[mask]
            
            # Points that escape
            diverged = np.abs(z) > 2.0
            
            # Update the mask for next iteration
            updated_mask = diverged & mask
            
            # Record the iteration count for points that escape in this iteration
            iterations[updated_mask] = iter_count + 1 - np.log(np.log(np.abs(z[updated_mask]))) / np.log(2)
            
            # Update the mask for next iteration
            mask[diverged] = False
            
            iter_count += 1
        
        # Transpose to match the image coordinates
        return iterations.T


class JuliaSet(FractalGenerator):
    """Julia set fractal generator."""
    
    def __init__(self, width, height, x_min, x_max, y_min, y_max, max_iter, color_scheme, c_real, c_imag):
        super().__init__(width, height, x_min, x_max, y_min, y_max, max_iter, color_scheme)
        self.c = complex(c_real, c_imag)
    
    def calculate(self):
        """Calculate the Julia set."""
        # Create a grid of complex numbers
        x = np.linspace(self.x_min, self.x_max, self.width)
        y = np.linspace(self.y_min, self.y_max, self.height)
        z = x[:, np.newaxis] + 1j * y[np.newaxis, :]
        
        # Initialize array for iterations
        iterations = np.zeros((self.width, self.height), dtype=np.float32)
        
        # Create a mask for points that are still being computed
        mask = np.ones_like(iterations, dtype=bool)
        
        # Iteration counter
        iter_count = 0
        
        # Iterate the Julia function z = z^2 + c
        while iter_count < self.max_iter and np.any(mask):
            # Check for timeout on regular intervals
            if iter_count % self.check_frequency == 0:
                self.check_timeout()
                
            z[mask] = z[mask] * z[mask] + self.c
            
            # Points that escape
            diverged = np.abs(z) > 2.0
            
            # Update the mask for next iteration
            updated_mask = diverged & mask
            
            # Record the iteration count for points that escape in this iteration
            iterations[updated_mask] = iter_count + 1 - np.log(np.log(np.abs(z[updated_mask]))) / np.log(2)
            
            # Update the mask for next iteration
            mask[diverged] = False
            
            iter_count += 1
        
        # Transpose to match the image coordinates
        return iterations.T


class BurningShip(FractalGenerator):
    """Burning Ship fractal generator."""
    
    def calculate(self):
        """Calculate the Burning Ship fractal."""
        # Create a grid of complex numbers
        x = np.linspace(self.x_min, self.x_max, self.width)
        y = np.linspace(self.y_min, self.y_max, self.height)
        c = x[:, np.newaxis] + 1j * y[np.newaxis, :]
        
        # Initialize array for iterations
        iterations = np.zeros((self.width, self.height), dtype=np.float32)
        
        # Initialize z
        z = np.zeros_like(c, dtype=np.complex64)
        
        # Create a mask for points that are still being computed
        mask = np.ones_like(iterations, dtype=bool)
        
        # Iteration counter
        iter_count = 0
        
        # Iterate the Burning Ship function z = (|Re(z)| + i|Im(z)|)^2 + c
        while iter_count < self.max_iter and np.any(mask):
            # Check for timeout on regular intervals
            if iter_count % self.check_frequency == 0:
                self.check_timeout()
                
            # Take absolute values of real and imaginary parts
            z_abs = np.abs(z.real) + 1j * np.abs(z.imag)
            
            # Square and add c
            z[mask] = z_abs[mask] * z_abs[mask] + c[mask]
            
            # Points that escape
            diverged = np.abs(z) > 2.0
            
            # Update the mask for next iteration
            updated_mask = diverged & mask
            
            # Record the iteration count for points that escape in this iteration
            iterations[updated_mask] = iter_count + 1 - np.log(np.log(np.abs(z[updated_mask]))) / np.log(2)
            
            # Update the mask for next iteration
            mask[diverged] = False
            
            iter_count += 1
        
        # Transpose to match the image coordinates
        return iterations.T


def generate_fractal_image(fractal_type, width, height, x_min, x_max, y_min, y_max, 
                          max_iter, color_scheme, c_real=None, c_imag=None):
    """
    Generate a fractal image based on the specified parameters.
    
    Args:
        fractal_type: Type of fractal ('mandelbrot', 'julia', 'burning_ship')
        width: Image width in pixels
        height: Image height in pixels
        x_min, x_max, y_min, y_max: Coordinates in the complex plane
        max_iter: Maximum number of iterations
        color_scheme: Matplotlib colormap name
        c_real, c_imag: Real and imaginary parts of c for Julia set (optional)
    
    Returns:
        A PIL Image object
    """
    if fractal_type == 'mandelbrot':
        generator = MandelbrotSet(width, height, x_min, x_max, y_min, y_max, max_iter, color_scheme)
    elif fractal_type == 'julia':
        # Default values for Julia set
        if c_real is None:
            c_real = -0.7
        if c_imag is None:
            c_imag = 0.27015
        generator = JuliaSet(width, height, x_min, x_max, y_min, y_max, max_iter, color_scheme, c_real, c_imag)
    elif fractal_type == 'burning_ship':
        generator = BurningShip(width, height, x_min, x_max, y_min, y_max, max_iter, color_scheme)
    else:
        raise ValueError(f"Unsupported fractal type: {fractal_type}")
    
    return generator.generate_image()


if __name__ == "__main__":
    # Test the fractal generator
    img = generate_fractal_image(
        fractal_type='mandelbrot',
        width=800,
        height=600,
        x_min=-2.5,
        x_max=1.0,
        y_min=-1.5,
        y_max=1.5,
        max_iter=100,
        color_scheme='viridis'
    )
    img.save('test_fractal.png') 