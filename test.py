import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class BinaryPerceptron:
    def __init__(self, n: int , P: int, seed: int):
        '''
        Initializations
        '''
        np.random.seed(seed)
        self.n = n
        self.P = P
        self.alpha = P/n
        self.seed = seed
        self.targets = np.random.choice([-1,1], size=P)
        self.weights = np.zeros(n)

    def init_config(self):
        '''
        Initial configuration of the objective matrix
        '''
        np.random.seed(self.seed)
        self.X = np.random.normal(loc = 0, scale = 1, size = (self.P, self.n))
        self.weights = np.random.choice([-1,1], size=self.n)

    def compute_cost(self):
        '''
        Define the cost function and computation
        '''
        self.pred = self.forward()
        wrong_bool = (self.pred * self.targets) < 0
        cost = wrong_bool.sum()
        return cost

    def compute_delta_cost(self, action):
        '''
        Compute delta cost of a given action efficiently
        '''
        # current pred
        current_pred = self.pred
        # delta predictions mathematically correct
        delta_pred = -2 * self.X[:, action] * self.weights[action]
        # derive new pred from the delta
        new_pred = current_pred + delta_pred.flatten()
        #current cost
        current_errors = (current_pred * self.targets) < 0
        current_cost = np.sum(current_errors)
        # new cost
        new_errors = (new_pred * self.targets) < 0
        new_cost = np.sum(new_errors)
        # compute delta
        delta = new_cost - current_cost
        
        return delta

    def accept_action(self, action):
        '''
        Update the internal states given the taken action
        '''
        delta_pred = (-2 * self.X[:, action] * self.weights[action]).flatten()
        self.pred = self.pred + delta_pred
        self.weights[action] = - self.weights[action]

    def propose_action(self):
        '''
        Propose a move based on some criteria
        '''
        index = np.random.choice(range(self.n), size=1)
        return index

    def copy(self):
        '''
        Copy the whole problem
        '''
        return deepcopy(self)

    def display(self):
        '''
        Display the current state
        '''
        pass
    
    def forward(self):
        '''
        Function that outputs the prediction in the current state
        '''
        intermediate = self.X @ self.weights
        return intermediate
    
    def visualize_landscape(self, weight_idx1=0, weight_idx2=1, resolution=50, weight_range=3):
        '''
        Visualize the 3D landscape of the cost function by varying two weight components
        
        Parameters:
        - weight_idx1, weight_idx2: indices of the two weights to vary
        - resolution: number of points along each axis
        - weight_range: range of weight values to explore (from -weight_range to +weight_range)
        '''
        if not hasattr(self, 'X'):
            print("Warning: init_config() not called. Initializing configuration...")
            self.init_config()
        
        # Create a grid of weight values
        w1_values = np.linspace(-weight_range, weight_range, resolution)
        w2_values = np.linspace(-weight_range, weight_range, resolution)
        W1, W2 = np.meshgrid(w1_values, w2_values)
        
        # Store original weights
        original_weights = self.weights.copy()
        
        # Initialize cost surface
        costs = np.zeros((resolution, resolution))
        
        # Compute cost for each combination of weight values
        for i, w1 in enumerate(w1_values):
            for j, w2 in enumerate(w2_values):
                # Set the two weights we're varying
                self.weights[weight_idx1] = w1
                self.weights[weight_idx2] = w2
                
                # Compute cost at this point
                costs[j, i] = self.compute_cost()
        
        # Restore original weights
        self.weights = original_weights
        
        # Create the 3D plot
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot the surface
        surf = ax.plot_surface(W1, W2, costs, cmap='viridis', 
                              alpha=0.8, linewidth=0, antialiased=True)
        
        # Add contour lines at the bottom
        ax.contour(W1, W2, costs, zdir='z', offset=np.min(costs)-1, 
                  cmap='viridis', alpha=0.5)
        
        # Mark the current position
        current_cost = self.compute_cost()
        ax.scatter([original_weights[weight_idx1]], 
                  [original_weights[weight_idx2]], 
                  [current_cost], 
                  color='red', s=100, label='Current Position')
        
        # Customize the plot
        ax.set_xlabel(f'Weight {weight_idx1}')
        ax.set_ylabel(f'Weight {weight_idx2}')
        ax.set_zlabel('Cost (Number of Misclassified Points)')
        ax.set_title(f'Binary Perceptron Cost Landscape\n(n={self.n}, P={self.P}, α={self.alpha:.2f})')
        
        # Add colorbar
        fig.colorbar(surf, shrink=0.5, aspect=20)
        
        # Add legend
        ax.legend()
        
        # Show statistics
        min_cost = np.min(costs)
        max_cost = np.max(costs)
        print(f"Landscape Statistics:")
        print(f"- Minimum cost: {min_cost}")
        print(f"- Maximum cost: {max_cost}")
        print(f"- Current cost: {current_cost}")
        print(f"- Cost range: {max_cost - min_cost}")
        
        plt.tight_layout()
        plt.show()
        
        return fig, ax, costs

# Example usage
if __name__ == "__main__":
    # Create a small perceptron problem for visualization
    perceptron = BinaryPerceptron(n=10, P=15, seed=42)
    perceptron.init_config()
    
    print("Initial cost:", perceptron.compute_cost())
    print("Weights shape:", perceptron.weights.shape)
    print("Data shape:", perceptron.X.shape)
    
    # Visualize the landscape
    fig, ax, costs = perceptron.visualize_landscape(weight_idx1=0, weight_idx2=1, 
                                                   resolution=30, weight_range=2)
    
    # You can also visualize different weight pairs
    # perceptron.visualize_landscape(weight_idx1=2, weight_idx2=3, resolution=30)