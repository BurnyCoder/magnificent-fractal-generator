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
        """Check if calculation has exceeded the timeout
        
        Returns:
            bool: True if timeout has occurred, False otherwise
        """
        elapsed = time.time() - self.start_time
        if elapsed > self.timeout:
            print(f"Fractal calculation timed out after {elapsed:.2f} seconds")
            return True
        return False
    
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
        
        # If calculation timed out and no data was returned, raise a TimeoutError
        if self.check_timeout() and data is None:
            elapsed = time.time() - start
            raise TimeoutError(f"Fractal calculation exceeded time limit of {self.timeout} seconds (elapsed: {elapsed:.2f}s)")
        
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


class Tricorn(FractalGenerator):
    """Tricorn (Mandelbar) fractal generator."""
    
    def calculate(self):
        """Calculate the Tricorn fractal."""
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
        
        # Iterate the Tricorn function z = z̄^2 + c (where z̄ is the complex conjugate of z)
        while iter_count < self.max_iter and np.any(mask):
            # Check for timeout on regular intervals
            if iter_count % self.check_frequency == 0:
                self.check_timeout()
                
            # Calculate the complex conjugate of z
            z_conj = np.conjugate(z)
            
            # Square conjugate and add c
            z[mask] = z_conj[mask] * z_conj[mask] + c[mask]
            
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


class MultibrotSet(FractalGenerator):
    """Multibrot Set fractal generator with adjustable power."""
    
    def __init__(self, width, height, x_min, x_max, y_min, y_max, max_iter, color_scheme, power=3):
        super().__init__(width, height, x_min, x_max, y_min, y_max, max_iter, color_scheme)
        self.power = power  # Default to cubic (power=3)
    
    def calculate(self):
        """Calculate the Multibrot set with specified power."""
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
        
        # Multibrot escape radius depends on the power
        escape_radius = max(2.0, pow(2.0, 1.0/(self.power-1)))
        
        # Iterate the Multibrot function z = z^power + c
        while iter_count < self.max_iter and np.any(mask):
            # Check for timeout on regular intervals
            if iter_count % self.check_frequency == 0:
                self.check_timeout()
                
            # Calculate z^power + c
            z[mask] = np.power(z[mask], self.power) + c[mask]
            
            # Points that escape
            diverged = np.abs(z) > escape_radius
            
            # Update the mask for next iteration
            updated_mask = diverged & mask
            
            # Record the iteration count for points that escape in this iteration
            iterations[updated_mask] = iter_count + 1 - np.log(np.log(np.abs(z[updated_mask]))) / np.log(self.power)
            
            # Update the mask for next iteration
            mask[diverged] = False
            
            iter_count += 1
        
        # Transpose to match the image coordinates
        return iterations.T


class Phoenix(FractalGenerator):
    """Phoenix fractal generator."""
    
    def __init__(self, width, height, x_min, x_max, y_min, y_max, max_iter, color_scheme, p_real=-0.5, p_imag=0.0):
        super().__init__(width, height, x_min, x_max, y_min, y_max, max_iter, color_scheme)
        self.p = complex(p_real, p_imag)  # Default parameter
    
    def calculate(self):
        """Calculate the Phoenix fractal."""
        # Create a grid of complex numbers
        x = np.linspace(self.x_min, self.x_max, self.width)
        y = np.linspace(self.y_min, self.y_max, self.height)
        c = x[:, np.newaxis] + 1j * y[np.newaxis, :]
        
        # Initialize arrays for current and previous values
        z = np.zeros_like(c, dtype=np.complex64)
        z_old = np.zeros_like(c, dtype=np.complex64)
        
        # Initialize array for iterations
        iterations = np.zeros((self.width, self.height), dtype=np.float32)
        
        # Create a mask for points that are still being computed
        mask = np.ones_like(iterations, dtype=bool)
        
        # Iteration counter
        iter_count = 0
        
        # Iterate the Phoenix function z_{n+1} = z_n^2 + c + p*z_{n-1}
        while iter_count < self.max_iter and np.any(mask):
            # Check for timeout on regular intervals
            if iter_count % self.check_frequency == 0:
                self.check_timeout()
                
            # Save temp for swapping
            z_temp = z.copy()
            
            # Phoenix formula
            z[mask] = z[mask] * z[mask] + c[mask] + self.p * z_old[mask]
            
            # Update old z value for next iteration
            z_old[mask] = z_temp[mask]
            
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


class Newton(FractalGenerator):
    """Newton fractal generator."""
    
    def __init__(self, width, height, x_min, x_max, y_min, y_max, max_iter, color_scheme):
        super().__init__(width, height, x_min, x_max, y_min, y_max, max_iter, color_scheme)
        # Roots of z^3 - 1 = 0
        self.roots = np.array([1, -0.5 + 0.866j, -0.5 - 0.866j], dtype=np.complex64)
        
    def calculate(self):
        """Calculate the Newton fractal for z^3 - 1 = 0."""
        # Create a grid of complex numbers
        x = np.linspace(self.x_min, self.x_max, self.width)
        y = np.linspace(self.y_min, self.y_max, self.height)
        z = x[:, np.newaxis] + 1j * y[np.newaxis, :]
        
        # Initialize array for iterations and root index
        iterations = np.zeros((self.width, self.height), dtype=np.float32)
        root_index = np.zeros_like(iterations, dtype=np.int8)
        
        # Create a mask for points that are still being computed
        mask = np.ones_like(iterations, dtype=bool)
        
        # Iteration counter
        iter_count = 0
        
        # Newton's method for finding roots of f(z) = z^3 - 1
        # z_{n+1} = z_n - f(z_n)/f'(z_n) = z_n - (z_n^3 - 1)/(3*z_n^2) = (2*z_n^3 + 1)/(3*z_n^2)
        while iter_count < self.max_iter and np.any(mask):
            # Check for timeout on regular intervals
            if iter_count % self.check_frequency == 0:
                self.check_timeout()
                
            # Newton's method formula for z^3 - 1
            z_squared = z * z
            z_cubed = z_squared * z
            z[mask] = (2 * z_cubed[mask] + 1) / (3 * z_squared[mask])
            
            # Check if we're close to any root
            for i, root in enumerate(self.roots):
                close_to_root = np.abs(z - root) < 1e-6
                # Record which root we're close to
                root_index[close_to_root & mask] = i + 1
                # Record the iteration count
                iterations[close_to_root & mask] = iter_count + 1
                # Update the mask
                mask[close_to_root] = False
                
            iter_count += 1
            
        # Create a more interesting visualization by combining root index and iteration count
        result = root_index.astype(np.float32) + iterations / (self.max_iter * 3)
        
        # Transpose to match the image coordinates
        return result.T


class SierpinskiCarpet(FractalGenerator):
    """Sierpinski Carpet fractal generator."""
    
    def __init__(self, width, height, x_min, x_max, y_min, y_max, max_iter, color_scheme, level=5):
        super().__init__(width, height, x_min, x_max, y_min, y_max, max_iter, color_scheme)
        self.level = min(int(level), 8)  # Limit maximum level to avoid excessive computation
        
    def calculate(self):
        """Calculate the Sierpinski Carpet fractal."""
        # Create a grid of points
        x = np.linspace(self.x_min, self.x_max, self.width)
        y = np.linspace(self.y_min, self.y_max, self.height)
        
        # Create the result array
        result = np.zeros((self.width, self.height), dtype=np.float32)
        
        # Normalize coordinates to [0, 1] range for easier computation
        x_norm = (x - self.x_min) / (self.x_max - self.x_min)
        y_norm = (y - self.y_min) / (self.y_max - self.y_min)
        
        # Check each point if it belongs to the carpet
        for i in range(self.width):
            # Check for timeout on regular intervals
            if i % self.check_frequency == 0:
                self.check_timeout()
                
            x_val = x_norm[i]
            for j in range(self.height):
                y_val = y_norm[j]
                
                # For each level, check if point is in the middle third
                belongs_to_carpet = True
                for level in range(1, self.level + 1):
                    divisor = 3 ** level
                    x_third = (x_val * divisor) % 3
                    y_third = (y_val * divisor) % 3
                    
                    # If both x and y are in the middle third (1.0-2.0), it's a hole
                    if 1.0 <= x_third < 2.0 and 1.0 <= y_third < 2.0:
                        belongs_to_carpet = False
                        # Store the level at which the point was removed
                        result[i, j] = level / self.level
                        break
                
                # If it belongs to the carpet after all levels, mark it
                if belongs_to_carpet:
                    result[i, j] = 1.0
        
        # Transpose to match the image coordinates
        return result.T


class LyapunovFractal(FractalGenerator):
    """Lyapunov fractal generator based on logistic map sequences."""
    
    def __init__(self, width, height, x_min, x_max, y_min, y_max, max_iter, color_scheme, sequence="AB"):
        super().__init__(width, height, x_min, x_max, y_min, y_max, max_iter, color_scheme)
        self.sequence = sequence if sequence else "AB"  # Default sequence is "AB"
        
    def calculate(self):
        """Calculate the Lyapunov fractal based on the sequence."""
        # Create a grid of points
        result = np.zeros((self.width, self.height), dtype=np.float32)
        
        # Expand sequence if it's too short
        seq = self.sequence
        if len(seq) < 2:
            seq = seq * 2
        
        seq_length = len(seq)
        
        # Iterate over the grid
        for i in range(self.width):
            # Check for timeout on regular intervals
            if i % self.check_frequency == 0:
                self.check_timeout()
                
            r_a = self.x_min + (self.x_max - self.x_min) * i / self.width
            
            for j in range(self.height):
                r_b = self.y_min + (self.y_max - self.y_min) * j / self.height
                
                # Skip invalid parameter ranges for the logistic map
                if r_a < 0 or r_a > 4 or r_b < 0 or r_b > 4:
                    result[i, j] = float('nan')
                    continue
                
                # Initial value for the logistic map
                x = 0.5
                
                # Calculate Lyapunov exponent
                lyapunov = 0.0
                
                # Discard transient iterations
                for _ in range(min(100, self.max_iter // 2)):
                    r = r_a if seq[_ % seq_length] == 'A' else r_b
                    x = r * x * (1 - x)
                
                # Calculate the Lyapunov exponent
                for n in range(self.max_iter):
                    r = r_a if seq[n % seq_length] == 'A' else r_b
                    derivative = r * (1 - 2 * x)
                    x = r * x * (1 - x)
                    
                    # Avoid log(0)
                    if derivative != 0:
                        lyapunov += np.log(abs(derivative))
                
                # Normalize the Lyapunov exponent
                lyapunov /= self.max_iter
                
                # Store the result - normalize for visualization
                if lyapunov <= 0:
                    # Stable (negative Lyapunov exponent)
                    result[i, j] = -lyapunov / 10  # Map negative values to 0-0.1 range
                else:
                    # Chaotic (positive Lyapunov exponent)
                    result[i, j] = 0.5 + lyapunov / 5  # Map positive values to 0.5-0.7 range
        
        # Transpose to match the image coordinates
        return result.T


class Buddhabrot(FractalGenerator):
    """
    Implementation of the Buddhabrot fractal, which is a rendering method for the Mandelbrot set
    that shows the density of escape trajectories.
    """
    def __init__(self, width, height, x_min, x_max, y_min, y_max, max_iter, color_scheme, samples=1000000):
        super().__init__(width, height, x_min, x_max, y_min, y_max, max_iter, color_scheme)
        # Number of random samples to use - adjust based on image size and timeout
        # For web UI, we need to be more conservative to avoid timeouts
        max_safe_samples = 100000 + (width * height) // 10  # Scale with image size
        self.samples = min(samples, max_safe_samples)
    
    def calculate(self):
        """Calculate the Buddhabrot fractal."""
        # Initialize histogram to count trajectory visits
        histogram = np.zeros((self.width, self.height), dtype=np.uint32)
        
        # Scale to convert from complex plane to pixel coordinates
        x_scale = self.width / (self.x_max - self.x_min)
        y_scale = self.height / (self.y_max - self.y_min)
        
        # Random sampling approach
        np.random.seed(42)  # For reproducibility
        
        # Counter for tracking progress
        processed = 0
        min_samples = 10000  # Minimum samples to process before we can return a partial result
        
        # Generate random complex points and track their trajectories
        for i in range(self.samples):
            # Check for timeout every 1000 samples to avoid slowing calculation
            if i % 1000 == 0 and self.check_timeout():
                # If we've processed enough samples, we can return a partial result
                if processed >= min_samples:
                    print(f"Buddhabrot calculation timed out but returning partial result with {processed} samples")
                    break
                else:
                    # Not enough samples, let generate_image handle the timeout
                    return None
                
            # Generate random point in complex plane
            c_real = np.random.uniform(self.x_min, self.x_max)
            c_imag = np.random.uniform(self.y_min, self.y_max)
            c = complex(c_real, c_imag)
            
            # Skip points that are in the main cardioid or period-2 bulb (optimization)
            # p = (c_real - 0.25)**2 + c_imag**2
            # if c_real <= p - 2 * p**2 + 0.25 or (c_real + 1)**2 + c_imag**2 <= 0.0625:
            #    continue
            
            # Track the trajectory
            z = complex(0, 0)
            trajectory = []
            escaped = False
            
            for j in range(self.max_iter):
                trajectory.append(z)
                z = z*z + c
                
                if abs(z) > 2.0:
                    escaped = True
                    break
            
            # Only count trajectories that escape
            if escaped:
                for point in trajectory:
                    x = int((point.real - self.x_min) * x_scale)
                    y = int((point.imag - self.y_min) * y_scale)
                    
                    # Check if point is within image bounds
                    if 0 <= x < self.width and 0 <= y < self.height:
                        histogram[x, y] += 1
            
            processed += 1
        
        # Normalize and scale the histogram for better visualization
        nonzero = histogram > 0
        if np.any(nonzero):
            # Apply log scaling to enhance detail
            result = np.zeros_like(histogram, dtype=float)
            result[nonzero] = np.log(histogram[nonzero])
            
            # Normalize to 0-1 range
            if np.max(result) > 0:
                result = result / np.max(result)
            
            return result
        else:
            return np.zeros((self.width, self.height), dtype=float)


def generate_fractal_image(fractal_type, width, height, x_min, x_max, y_min, y_max, 
                          max_iter, color_scheme, c_real=None, c_imag=None):
    """
    Generate a fractal image based on the specified parameters.
    
    Args:
        fractal_type: Type of fractal ('mandelbrot', 'julia', 'burning_ship', etc.)
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
    elif fractal_type == 'tricorn':
        generator = Tricorn(width, height, x_min, x_max, y_min, y_max, max_iter, color_scheme)
    elif fractal_type == 'newton':
        generator = Newton(width, height, x_min, x_max, y_min, y_max, max_iter, color_scheme)
    elif fractal_type == 'multibrot':
        # Default to cubic Multibrot
        power = float(c_real) if c_real is not None else 3
        generator = MultibrotSet(width, height, x_min, x_max, y_min, y_max, max_iter, color_scheme, power=power)
    elif fractal_type == 'phoenix':
        # Default values for Phoenix parameters
        p_real = c_real if c_real is not None else -0.5
        p_imag = c_imag if c_imag is not None else 0.0
        generator = Phoenix(width, height, x_min, x_max, y_min, y_max, max_iter, color_scheme, p_real, p_imag)
    elif fractal_type == 'sierpinski_carpet':
        # Default level for Sierpinski Carpet
        level = int(c_real) if c_real is not None else 5
        generator = SierpinskiCarpet(width, height, x_min, x_max, y_min, y_max, max_iter, color_scheme, level=level)
    elif fractal_type == 'lyapunov':
        # Default sequence for Lyapunov fractal
        sequence = c_real if isinstance(c_real, str) else "AB"
        generator = LyapunovFractal(width, height, x_min, x_max, y_min, y_max, max_iter, color_scheme, sequence=sequence)
    elif fractal_type == 'buddhabrot':
        # Use c_real as the number of samples
        samples = int(c_real) if c_real is not None else 1000000
        generator = Buddhabrot(width, height, x_min, x_max, y_min, y_max, max_iter, color_scheme, samples=samples)
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