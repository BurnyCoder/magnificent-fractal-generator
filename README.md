# Magnificent Fractal Art Generator

A beautiful interactive fractal art generator with a clean, modern web interface. Create stunning visual art through the mathematical beauty of fractals.

![Fractal examples](fractals.png)

## Features

- Interactive fractal generator with real-time updates
- Multiple fractal types (Mandelbrot, Julia sets, Burning Ship, Tricorn, Newton, Multibrot, Phoenix, Sierpinski Carpet, Lyapunov, Buddhabrot)
- Customizable color palettes and parameters
- Ability to zoom, pan, and explore fractals in high detail
- Save and share your creations
- Responsive design that works on desktop and mobile

### Mathematical Background

Fractals are created through iterative equations applied to each point in the complex plane:

- **Mandelbrot Set**: z → z² + c, where c is the point's coordinates
- **Julia Set**: z → z² + c, where c is a fixed complex parameter
- **Burning Ship**: z → (|Re(z)| + i|Im(z)|)² + c
- **Tricorn (Mandelbar)**: z → z̅² + c, where z̅ is the complex conjugate
- **Newton Fractal**: z → z - f(z)/f'(z), using Newton's method to find roots of f(z)
- **Multibrot Set**: z → z^n + c, where n > 2 is the power parameter (generalizes the Mandelbrot)
- **Phoenix Fractal**: z_{n+1} → z_n² + c + p*z_{n-1}, where p is a complex parameter
- **Sierpinski Carpet**: A recursive fractal created by repeatedly removing the middle ninth of squares
- **Lyapunov Fractal**: Visualizes the stability (Lyapunov exponent) of the logistic map x_{n+1} = r*x_n*(1-x_n) under different parameter sequences
- **Buddhabrot**: A special rendering technique that plots the trajectories of points that escape the Mandelbrot set rather than the points themselves, creating an image that resembles a meditating Buddha

## Installation

1. Clone this repository
2. Create a virtual environment (recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Run the application:
   ```
   python app.py
   ```
5. Open your browser and navigate to `http://localhost:5000`

## Usage Guide

### Creating Fractals

1. Select a fractal type (Mandelbrot, Julia, Burning Ship, Tricorn, Newton, Multibrot, Phoenix, Sierpinski Carpet, Lyapunov, or Buddhabrot)
2. Adjust parameters as desired:
   - Color scheme
   - Maximum iterations
   - Resolution
   - Complex plane coordinates
   - Type-specific parameters (power, complex parameters, recursion level, sequence, etc.)
3. Click "Generate" to create your fractal
4. Use the zoom and pan controls to explore interesting areas

### Saving and Sharing

- Click "Save" to store your creation in your gallery (requires login)
- Click "Download" to save the image to your computer
- Visit the Gallery to see fractals from other users

## Advanced Features

### Julia Set Exploration

The Julia Set requires a complex parameter c. Try these interesting values:
- c = -0.7 + 0.27015i (default)
- c = -0.8 + 0.156i
- c = -0.4 + 0.6i
- c = 0.285 + 0.01i

## License

MIT License 