"""
Created on Fri Dec 26 22:01:45 2025

@author: sid

F1 Aerodynamic Simulation: ML/AI Analysis of CFD/Wind Tunnel Data
Based on 2026 F1 Technical Regulations

This project demonstrates:
1. Synthetic aerodynamic data generation
2. Machine learning models for drag/downforce prediction
3. Design optimization using neural networks
4. Visualization of aerodynamic performance
"""

import numpy as np
import pandas as pd
import seaborn as sns
import warnings as w
w.filterwarnings('ignore')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from scipy.optimize import differential_evolution
from advance_visualisation import VisualizationEngine
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class F1AerodynamicSimulator:
    """
    Simulate F1 Aerodynamic data based on 2026 Technical Regulations
    
    Key aerodynamic components from regulations:
    - Front Wing (RV-FW-PROFILES): Article 3.10
    - Rear Wing (RV-RW-PROFILES): Article 3.11
    - Floor (RV-FLOOR-BODY): Article 3.5
    - Diffuser and underbody flow
    """
    def __init__(self, n_samples=1000):
        self.n_samples = n_samples
        self.data = None
        
    def generate_synthetic_data(self):
        """
        Generate synthetic CFD/wind tunnel data for F1 Aerodynamics
        
        Design parameters based on 2026 regulations:
            - Front wing angle: affects front downforce and drag
            - Rear wing angle: affects rear downforce and drag
            - Ride height: affects floor performance
            - Rake angle: pitch of the car
            - Floor edge shape: venturi effect
        """
        np.random.seed(42)
        
        #Design parameters inut
        front_wing_angle = np.random.uniform(15,35,self.n_samples)
        rear_wing_angle = np.random.uniform(5,25,self.n_samples)
        ride_height_front = np.random.uniform(20,50,self.n_samples)
        ride_height_rear = np.random.uniform(40,80,self.n_samples)
        floor_edge_radius = np.random.uniform(25,50,self.n_samples)
        sidepod_inlet_area = np.random.uniform(0.06, 0.08, self.n_samples)
        diffuser_angle = np.random.uniform(12,18,self.n_samples)
        speed = np.random.uniform(50, 350, self.n_samples) 
        
        rake_angle = np.arctan((ride_height_rear - ride_height_front) / 3000) * 180/np.pi
        
        # Drag coefficient (Cd) - increases with wing angles and speed
        base_drag = 0.65
        cd = (base_drag +
              0.008 * front_wing_angle +
              0.012 * rear_wing_angle +
              0.002 * np.abs(rake_angle) + 
              0.001 * (speed/100) ** 2 + 
              np.random.normal(0, 0.02, self.n_samples))
        
        base_downforce = 2.5
        cl = (base_downforce +
              0.05 * front_wing_angle + 
              0.08 * rear_wing_angle + 
              0.03 * rake_angle +
              0.02 * (50 - floor_edge_radius) / 10 +
              0.15 * (80 - ride_height_rear) / 40 +
              np.random.normal(0, 0.08, self.n_samples))
        
        #L/D ratio (efficiency)
        ld_ratio = cl/cd
        
        balance = 0.45 + 0.01 * (front_wing_angle - 25) - 0.005 * (rear_wing_angle - 15)
        balance = np.clip(balance, 0.40, 0.50)
        
        self.data = pd.DataFrame({
            'front_wing_angle': front_wing_angle,
            'rear_wing_angle': rear_wing_angle,
            'ride_height_front': ride_height_front,
            'ride_height_rear': ride_height_rear,
            'rake_angle': rake_angle,
            'floor_edge_radius': floor_edge_radius,
            'sidepod_inlet_area': sidepod_inlet_area,
            'diffuser_angle': diffuser_angle,
            'speed': speed,
            'drag_coefficient': cd,
            'downforce_coefficient': cl,
            'ld_ratio': ld_ratio,
            'balance': balance
        })
        return self.data
    
    def get_features_and_targets(self):
        """Extract features (X) and targets (y) for ML models"""
        feature_cols = ['front_wing_angle', 'rear_wing_angle', 'ride_height_front', 
                       'ride_height_rear', 'rake_angle', 'floor_edge_radius',
                       'sidepod_inlet_area', 'diffuser_angle', 'speed']
        
        X = self.data[feature_cols]
        y_drag = self.data['drag_coefficient']
        y_downforce = self.data['downforce_coefficient']
        y_ld = self.data['ld_ratio']
        y_balance = self.data['balance']
        
        return X, y_drag, y_downforce, y_ld, y_balance
    

class AerodynamicMLAnalyzer:
    """
    Machine Learning analyzer for aerodynamic performance
    """
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        
    def train_models(self, X, y_dict):
        """
        Train multiple ML models for different aerodynamic targets
        
        Args:
            X: Feature matrix
            y_dict: Dictionary of targets {'drag': y_drag, 'downforce': y_downforce, etc.}
        """
        results = {}
        for target_name, y in y_dict.items():
            print(f"\n{'='*60}")
            print(f"Training models for: {target_name.upper()}")
            print('='*60)
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42)
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            self.scalers[target_name] = scaler
            
            models = {
                'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
                'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
                'Neural Network': MLPRegressor(hidden_layer_sizes=(64, 32, 16), max_iter=1000, random_state=42)
            }
            
            target_results = {}
            for name, model in models.items():
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                target_results[name] = {
                    'model': model,
                    'mse': mse,
                    'mae': mae,
                    'r2': r2,
                    'predictions': y_pred,
                    'actual': y_test
                }
                
                print(f"\n{name}:")
                print(f"  MSE: {mse:.6f}")
                print(f"  MAE: {mae:.6f}")
                print(f"  R²:  {r2:.6f}")
  
            best_model_name = max(target_results, key=lambda x: target_results[x]['r2'])
            self.models[target_name] = target_results[best_model_name]['model']
            
            # Feature importance (for tree-based models)
            if isinstance(self.models[target_name], (RandomForestRegressor, GradientBoostingRegressor)):
                self.feature_importance[target_name] = self.models[target_name].feature_importances_
            
            results[target_name] = target_results
        
        return results
    
    def optimize_design(self, X_columns, target = 'ld_ratio', n_iterations = 50):
        """
        Optimise aerodynamic design using trained Models
        
        Target: Maximise L/D ratio or other performance metric
        Constraints: Based on 2026 F1 Technical Regulations
        
        """
        print(f"\n{'='*60}")
        print(f"OPTIMIZING DESIGN FOR: {target.upper()}")
        print('='*60)
        
        bounds = [
            (15, 35),   # front_wing_angle
            (5, 25),    # rear_wing_angle
            (20, 50),   # ride_height_front
            (40, 80),   # ride_height_rear
            (12, 18),   # rake_angle (derived)
            (25, 50),   # floor_edge_radius (Article 3.5)
            (0.06, 0.08), # sidepod_inlet_area
            (12, 18),   # diffuser_angle
            (200, 300)  # speed (typical race speed range)
            ]
        
        def objective(params):
            """Objective function to minimize (negative of target)"""
            X_input = np.array(params).reshape(1,-1)
            X_scaled = self.scalers[target].transform(X_input)
            prediction = self.models[target].predict(X_scaled)[0]
            #  Minimize negative (maximize target)
            return -prediction
        
        result = differential_evolution(
            objective,
            bounds,
            maxiter = n_iterations,
            seed = 42,
            disp = True
            )
        
        optimal_params = dict(zip(X_columns, result.x))
        optimal_value = -result.fun
        
        print(f"\nOptimal Design Parameters:")
        print("-" * 60)
        for param, value in optimal_params.items():
            print(f"  {param:25s}: {value:8.3f}")
        print(f"\nOptimized {target}: {optimal_value:.4f}")
        
        return optimal_params, optimal_value
    
    def visualise_results(simulator, analyzer, results):
        """Create comprehensive visualizations"""
        fig = plt.figure(figsize=(16, 12))
        
        #1. Model Performance comparison
        ax1 = plt.subplot(2,3,1)
        targets = list(results.keys())
        models = list(results[targets[0]].keys())
        r2_scores = np.array([[results[target][model]['r2'] for model in models] for target in targets])
        x = np.arange(len(targets))
        width = 0.25
        for i, model in enumerate(models):
            ax1.bar(x + i*width, r2_scores[:, i], width, label=model)
    
        ax1.set_xlabel('Target Variable')
        ax1.set_ylabel('R² Score')
        ax1.set_title('Model Performance Comparison')
        ax1.set_xticks(x + width)
        ax1.set_xticklabels(targets, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        #2. Actual vs Predicted (L/D Ratio)
        ax2 = plt.subplot(2,3,2)
        best_model = max(results['ld_ratio'], key=lambda x: results['ld_ratio'][x]['r2'])
        actual = results['ld_ratio'][best_model]['actual']
        predicted = results['ld_ratio'][best_model]['predictions']
    
        ax2.scatter(actual, predicted, alpha=0.5)
        ax2.plot([actual.min(), actual.max()], [actual.min(), actual.max()],'r--', lw=2, label='Perfect Prediction')
        ax2.set_xlabel('Actual L/D Ratio')
        ax2.set_ylabel('Predicted L/D Ratio')
        ax2.set_title(f'L/D Prediction ({best_model})')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        #3. Feature Importance
        ax3 = plt.subplot(2,3,3)
        if 'ld_ratio' in analyzer.feature_importance:
            importance = analyzer.feature_importance['ld_ratio']
        features = simulator.data.columns[:len(importance)]
        sorted_idx = np.argsort(importance)
        
        ax3.barh(range(len(importance)), importance[sorted_idx])
        ax3.set_yticks(range(len(importance)))
        ax3.set_yticklabels([features[i] for i in sorted_idx])
        ax3.set_xlabel('Feature Importance')
        ax3.set_title('L/D Ratio Feature Importance')
        ax3.grid(True, alpha=0.3)
        
        #4. Drag vs Downforce Trade-off
        ax4 = plt.subplot(2, 3, 4)
        scatter = ax4.scatter(simulator.data['drag_coefficient'], 
                         simulator.data['downforce_coefficient'],
                         c=simulator.data['ld_ratio'], 
                         cmap='viridis', alpha=0.6)
        ax4.set_xlabel('Drag Coefficient (Cd)')
        ax4.set_ylabel('Downforce Coefficient (Cl)')
        ax4.set_title('Drag vs Downforce Trade-off')
        plt.colorbar(scatter, ax=ax4, label='L/D Ratio')
        ax4.grid(True, alpha=0.3)
        
        #5. Wing Angle Effects
        ax5 = plt.subplot(2, 3, 5)
        ax5.scatter(simulator.data['rear_wing_angle'], 
               simulator.data['downforce_coefficient'],
               c=simulator.data['drag_coefficient'], 
               cmap='coolwarm', alpha=0.6)
        ax5.set_xlabel('Rear Wing Angle (degrees)')
        ax5.set_ylabel('Downforce Coefficient')
        ax5.set_title('Rear Wing Impact on Downforce')
        ax5.grid(True, alpha=0.3)
        
        #6. Ride Height vs Performance
        ax6 = plt.subplot(2, 3, 6)
        ax6.scatter(simulator.data['ride_height_rear'], 
               simulator.data['ld_ratio'],
               c=simulator.data['speed'], 
               cmap='plasma', alpha=0.6)
        ax6.set_xlabel('Rear Ride Height (mm)')
        ax6.set_ylabel('L/D Ratio')
        ax6.set_title('Ground Effect: Ride Height Impact')
        ax6.grid(True, alpha=0.3)
    
        plt.tight_layout()
        plt.savefig('f1_aerodynamic_analysis.png', dpi=300, bbox_inches='tight')
        print("\nVisualization saved as 'f1_aerodynamic_analysis.png'")
        plt.show()
        
def main():
    """ Main Execution Function"""
    
    print("="*70)
    print("F1 AERODYNAMIC SIMULATION: ML/AI ANALYSIS")
    print("Based on 2026 FIA Formula 1 Technical Regulations")
    print("="*70)
    
    # 1. Generate synthetic aerodynamic data
    print("\n[1] Generating synthetic CFD/Wind Tunnel data...")
    simulator = F1AerodynamicSimulator(n_samples=1000)
    data = simulator.generate_synthetic_data()
    print(f"Generated {len(data)} aerodynamic configurations")
    print("\nSample data:")
    print(data.head())
    
    # 2. Prepare data for ML
    print("\n[2] Preparing data for machine learning...")
    X, y_drag, y_downforce, y_ld, y_balance = simulator.get_features_and_targets()
    
    y_dict = {
        'drag': y_drag,
        'downforce': y_downforce,
        'ld_ratio': y_ld,
        'balance': y_balance
    }
    
    # 3. Train ML models
    print("\n[3] Training machine learning models...")
    analyzer = AerodynamicMLAnalyzer()
    results = analyzer.train_models(X, y_dict)
    
    # 4. Optimize design
    print("\n[4] Optimizing aerodynamic design...")
    optimal_design, optimal_ld = analyzer.optimize_design(
        X.columns, 
        target='ld_ratio',
        n_iterations=30
    )
    
    # 5. Visualize results using the new visualization module
    print("\n[5] Creating visualizations...")
    VisualizationEngine.visualise_results_improved(simulator, analyzer, results, optimal_design)
    
    # 6. Summary report
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE - SUMMARY")
    print("="*70)
    print(f"\n✓ Best L/D Ratio achieved: {optimal_ld:.4f}")
    print("\nKey Findings:")
    print("  • Machine learning models successfully predict aerodynamic performance")
    print("  • Optimization identifies ideal setup within 2026 regulations")
    print("  • Trade-offs between drag and downforce clearly identified")
    print("  • Ground effect (ride height) significantly impacts performance")
    print("  • Rear wing angle is the most critical aerodynamic parameter")
    print("\nRecommendations:")
    print("  • Focus on rear wing optimization for maximum L/D ratio")
    print("  • Minimize rear ride height while maintaining rake angle stability")
    print("  • Balance front/rear downforce for optimal handling characteristics")
    print("  • Monitor wing angle trade-offs across different track conditions")
    
    return simulator, analyzer, results, optimal_design
        
if __name__ == "__main__":
    simulator, analyzer, results, optimal_design = main()        