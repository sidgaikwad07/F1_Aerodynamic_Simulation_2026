#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  5 21:45:33 2026

@author: sid

F1 Aerodynamic Comparison Analysis
2025 vs 2026 Technical Regulations
"""

from aero_simulation_2026 import F1AerodynamicSimulator, AerodynamicMLAnalyzer
from comparison_visualisation_2025_vs_2026_analysis import ComparisonVisualizationEngine

class F1AerodynamicSimulator2025Real(F1AerodynamicSimulator):
    """2025: Peak downforce era with heavy ground effect dependency"""
    
    def generate_synthetic_data(self):
        """Generate 2025 data - PEAK DOWNFORCE ERA"""
        import numpy as np
        import pandas as pd
        
        np.random.seed(42)
        
        # 2025 REGULATIONS: Peak downforce, heavy ground effect
        front_wing_angle = np.random.uniform(18, 35, self.n_samples)
        rear_wing_angle = np.random.uniform(8, 25, self.n_samples)
        ride_height_front = np.random.uniform(20, 45, self.n_samples)
        ride_height_rear = np.random.uniform(35, 70, self.n_samples)
        floor_edge_radius = np.random.uniform(20, 45, self.n_samples)
        sidepod_inlet_area = np.random.uniform(0.068, 0.082, self.n_samples)
        diffuser_angle = np.random.uniform(13, 19, self.n_samples)
        speed = np.random.uniform(50, 350, self.n_samples)
        
        rake_angle = np.arctan((ride_height_rear - ride_height_front) / 3000) * 180/np.pi
        
        # 2025: HIGH DOWNFORCE baseline
        base_drag_2025 = 0.78  # Moderate-high drag (current era)
        cd = (base_drag_2025 +
              0.010 * front_wing_angle +
              0.014 * rear_wing_angle +
              0.0025 * np.abs(rake_angle) + 
              0.0012 * (speed/100) ** 2 + 
              np.random.normal(0, 0.02, self.n_samples))
        
        # 2025: PEAK DOWNFORCE - Heavy ground effect dominance
        base_downforce_2025 = 3.2  # Very high baseline
        cl = (base_downforce_2025 +
              0.055 * front_wing_angle +
              0.090 * rear_wing_angle +
              0.04 * rake_angle +
              0.035 * (50 - floor_edge_radius) / 10 +  # Ground effect VERY dominant
              0.20 * (80 - ride_height_rear) / 40 +    # Ride height VERY critical
              np.random.normal(0, 0.10, self.n_samples))
        
        # 2025: Very high downforce ceiling (peak era)
        cl = np.clip(cl, 4.8, 7.2)
        
        ld_ratio = cl / cd
        
        balance = 0.46 + 0.011 * (front_wing_angle - 25) - 0.0055 * (rear_wing_angle - 18)
        balance = np.clip(balance, 0.42, 0.52)
        
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


class F1AerodynamicSimulator2026Real(F1AerodynamicSimulator):
    """2026: New regulation era with REDUCED downforce (-30%), active aero, lighter cars"""
    
    def generate_synthetic_data(self):
        """Generate 2026 data - REDUCED DOWNFORCE ERA with active aero"""
        import numpy as np
        import pandas as pd
        
        np.random.seed(42)
        
        # 2026 REGULATIONS: 
        # - Downforce reduced by ~30%
        # - Drag reduced significantly with active aero
        # - Smaller, lighter cars
        # - Narrower tyres
        # - Active aerodynamics for overtaking
        
        front_wing_angle = np.random.uniform(12, 28, self.n_samples)  # Narrower range
        rear_wing_angle = np.random.uniform(5, 22, self.n_samples)    # Narrower range
        ride_height_front = np.random.uniform(22, 48, self.n_samples) # Slightly higher
        ride_height_rear = np.random.uniform(38, 72, self.n_samples)  # Slightly higher
        floor_edge_radius = np.random.uniform(25, 50, self.n_samples) # Less aggressive
        sidepod_inlet_area = np.random.uniform(0.060, 0.075, self.n_samples)  # Smaller
        diffuser_angle = np.random.uniform(11, 17, self.n_samples)    # Less aggressive
        speed = np.random.uniform(50, 350, self.n_samples)
        
        rake_angle = np.arctan((ride_height_rear - ride_height_front) / 3000) * 180/np.pi
        
        # 2026: LOWER DRAG with active aero systems
        # Active aero reduces drag on straights significantly
        base_drag_2026 = 0.62  # SIGNIFICANTLY LOWER (-20% vs 2025)
        cd = (base_drag_2026 +
              0.007 * front_wing_angle +    # Less sensitive (narrower range)
              0.010 * rear_wing_angle +     # Less sensitive
              0.0018 * np.abs(rake_angle) + 
              0.0008 * (speed/100) ** 2 + 
              np.random.normal(0, 0.02, self.n_samples))
        
        # 2026: REDUCED DOWNFORCE (-30% as per FIA requirement)
        # But shift from ground effect to active aero blend
        base_downforce_2026 = 2.24  # 30% LESS than 2025 (3.2 * 0.70 = 2.24)
        cl = (base_downforce_2026 +
              0.040 * front_wing_angle +    # Less responsive
              0.062 * rear_wing_angle +     # Less responsive
              0.025 * rake_angle +
              0.022 * (50 - floor_edge_radius) / 10 +  # Ground effect less dominant
              0.14 * (80 - ride_height_rear) / 40 +    # Ride height less critical
              np.random.normal(0, 0.08, self.n_samples))
        
        # 2026: Lower downforce ceiling (-30% as mandated)
        cl = np.clip(cl, 3.4, 5.0)  # Down from 4.8-7.2 in 2025
        
        ld_ratio = cl / cd
        
        # Balance window widens in 2026 (easier to balance with less aero)
        balance = 0.47 + 0.009 * (front_wing_angle - 20) - 0.0045 * (rear_wing_angle - 15)
        balance = np.clip(balance, 0.43, 0.53)
        
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


def main_comparison():
    """Main execution for realistic 2025 vs 2026 comparison"""
    
    print("="*80)
    print("F1 AERODYNAMIC COMPARISON: 2025 vs 2026 TECHNICAL REGULATIONS")
    print("Based on FIA Official Requirements")
    print("="*80)
    
    # ==================== 2025 ANALYSIS ====================
    print("\n" + "="*80)
    print("[2025 ANALYSIS] PEAK DOWNFORCE ERA")
    print("Heavy ground effect, maximum downforce, high drag")
    print("="*80)
    
    simulator_2025 = F1AerodynamicSimulator2025Real(n_samples=1000)
    data_2025 = simulator_2025.generate_synthetic_data()
    print(f"✓ Generated {len(data_2025)} aerodynamic configurations for 2025")
    print(f"\n  2025 AERODYNAMIC CHARACTERISTICS:")
    print(f"  • Drag range: {data_2025['drag_coefficient'].min():.3f} - {data_2025['drag_coefficient'].max():.3f}")
    print(f"  • Drag average: {data_2025['drag_coefficient'].mean():.3f}")
    print(f"  • Downforce range: {data_2025['downforce_coefficient'].min():.3f} - {data_2025['downforce_coefficient'].max():.3f}")
    print(f"  • Downforce average: {data_2025['downforce_coefficient'].mean():.3f}")
    print(f"  • L/D range: {data_2025['ld_ratio'].min():.3f} - {data_2025['ld_ratio'].max():.3f}")
    print(f"  • L/D average: {data_2025['ld_ratio'].mean():.3f}")
    
    print("\n[2025] Preparing data for machine learning...")
    X_2025, y_drag_2025, y_downforce_2025, y_ld_2025, y_balance_2025 = simulator_2025.get_features_and_targets()
    
    y_dict_2025 = {
        'drag': y_drag_2025,
        'downforce': y_downforce_2025,
        'ld_ratio': y_ld_2025,
        'balance': y_balance_2025
    }
    
    print("[2025] Training machine learning models...")
    analyzer_2025 = AerodynamicMLAnalyzer()
    results_2025 = analyzer_2025.train_models(X_2025, y_dict_2025)
    
    print("\n[2025] Optimizing aerodynamic design...")
    optimal_2025, optimal_ld_2025 = analyzer_2025.optimize_design(
        X_2025.columns, 
        target='ld_ratio',
        n_iterations=50
    )
    print(f"✓ 2025 Best L/D Ratio: {optimal_ld_2025:.4f}")
    
    # ==================== 2026 ANALYSIS ====================
    print("\n" + "="*80)
    print("[2026 ANALYSIS] NEW REGULATION ERA")
    print("Reduced downforce (-30%), lower drag, active aero, lighter cars")
    print("="*80)
    
    simulator_2026 = F1AerodynamicSimulator2026Real(n_samples=1000)
    data_2026 = simulator_2026.generate_synthetic_data()
    print(f"✓ Generated {len(data_2026)} aerodynamic configurations for 2026")
    print(f"\n  2026 AERODYNAMIC CHARACTERISTICS:")
    print(f"  • Drag range: {data_2026['drag_coefficient'].min():.3f} - {data_2026['drag_coefficient'].max():.3f}")
    print(f"  • Drag average: {data_2026['drag_coefficient'].mean():.3f}")
    print(f"  • Downforce range: {data_2026['downforce_coefficient'].min():.3f} - {data_2026['downforce_coefficient'].max():.3f}")
    print(f"  • Downforce average: {data_2026['downforce_coefficient'].mean():.3f}")
    print(f"  • L/D range: {data_2026['ld_ratio'].min():.3f} - {data_2026['ld_ratio'].max():.3f}")
    print(f"  • L/D average: {data_2026['ld_ratio'].mean():.3f}")
    
    print("\n[2026] Preparing data for machine learning...")
    X_2026, y_drag_2026, y_downforce_2026, y_ld_2026, y_balance_2026 = simulator_2026.get_features_and_targets()
    
    y_dict_2026 = {
        'drag': y_drag_2026,
        'downforce': y_downforce_2026,
        'ld_ratio': y_ld_2026,
        'balance': y_balance_2026
    }
    
    print("[2026] Training machine learning models...")
    analyzer_2026 = AerodynamicMLAnalyzer()
    results_2026 = analyzer_2026.train_models(X_2026, y_dict_2026)
    
    print("\n[2026] Optimizing aerodynamic design...")
    optimal_2026, optimal_ld_2026 = analyzer_2026.optimize_design(
        X_2026.columns, 
        target='ld_ratio',
        n_iterations=50
    )
    print(f"✓ 2026 Best L/D Ratio: {optimal_ld_2026:.4f}")
    
    # COMPARISON VISUALIZATION 
    print("\n" + "="*80)
    print("[COMPARISON] Creating comprehensive comparison visualizations...")
    print("="*80)
    
    ComparisonVisualizationEngine.visualise_2025_vs_2026(
        simulator_2025, analyzer_2025, results_2025, optimal_2025,
        simulator_2026, analyzer_2026, results_2026, optimal_2026
    )
    
    # ==================== DETAILED COMPARISON REPORT ====================
    print("\n" + "="*80)
    print("DETAILED PARAMETER COMPARISON: 2025 vs 2026")
    print("="*80)
    
    print("\n" + "="*50)
    print("2025 OPTIMAL CONFIGURATION (PEAK DOWNFORCE ERA):")
    print("="*50)
    for param, value in optimal_2025.items():
        if param in ['front_wing_angle', 'rear_wing_angle', 'rake_angle', 'diffuser_angle']:
            print(f"  {param.replace('_', ' ').upper():30s}: {value:8.3f}°")
        else:
            print(f"  {param.replace('_', ' ').upper():30s}: {value:8.3f}")
    print(f"  {'OPTIMIZED L/D RATIO':30s}: {optimal_ld_2025:8.4f}")
    
    print("\n" + "="*50)
    print("2026 OPTIMAL CONFIGURATION (NEW REGULATION ERA):")
    print("="*50)
    for param, value in optimal_2026.items():
        if param in ['front_wing_angle', 'rear_wing_angle', 'rake_angle', 'diffuser_angle']:
            print(f"  {param.replace('_', ' ').upper():30s}: {value:8.3f}°")
        else:
            print(f"  {param.replace('_', ' ').upper():30s}: {value:8.3f}")
    print(f"  {'OPTIMIZED L/D RATIO':30s}: {optimal_ld_2026:8.4f}")
    
    print("\n" + "="*50)
    print("PARAMETER CHANGES (2025 → 2026):")
    print("="*50)
    for param in optimal_2025.keys():
        if param in optimal_2026:
            change = optimal_2026[param] - optimal_2025[param]
            pct_change = (change / abs(optimal_2025[param])) * 100 if optimal_2025[param] != 0 else 0
            unit = "°" if param in ['front_wing_angle', 'rear_wing_angle', 'rake_angle', 'diffuser_angle'] else ""
            print(f"  {param.replace('_', ' ').upper():30s}: {change:+8.3f}{unit} ({pct_change:+6.2f}%)")
    
    l_d_change = optimal_ld_2026 - optimal_ld_2025
    l_d_pct = (l_d_change / optimal_ld_2025) * 100
    print(f"  {'L/D RATIO CHANGE':30s}: {l_d_change:+8.4f} ({l_d_pct:+6.2f}%)")
    
    # REALISTIC SUMMARY REPORT
    print("\n" + "="*80)
    print("COMPARISON ANALYSIS COMPLETE - EXECUTIVE SUMMARY")
    print("="*80)
    
    drag_2025_avg = data_2025['drag_coefficient'].mean()
    drag_2026_avg = data_2026['drag_coefficient'].mean()
    drag_change = ((drag_2026_avg - drag_2025_avg) / drag_2025_avg) * 100
    
    df_2025_avg = data_2025['downforce_coefficient'].mean()
    df_2026_avg = data_2026['downforce_coefficient'].mean()
    downforce_change = ((df_2026_avg - df_2025_avg) / df_2025_avg) * 100
    
    efficiency_improvement = ((optimal_ld_2026 - optimal_ld_2025) / optimal_ld_2025) * 100
    
    print(f"\n📊 AERODYNAMIC PERFORMANCE CHANGES (As Per FIA Requirements):")
    print(f"   • Drag Coefficient: {drag_2025_avg:.4f} (2025) → {drag_2026_avg:.4f} (2026)")
    print(f"     Change: {drag_change:+.2f}% {'✅ REDUCED' if drag_change < 0 else '⚠️ INCREASED'}")
    print(f"   • Downforce Coefficient: {df_2025_avg:.4f} (2025) → {df_2026_avg:.4f} (2026)")
    print(f"     Change: {downforce_change:+.2f}% {'✅ AS MANDATED' if -35 < downforce_change < -25 else '⚠️ CHECK'}")
    print(f"   • L/D Efficiency: {efficiency_improvement:+.2f}%")
    
    print(f"\n🎯 OPTIMAL DESIGN EVOLUTION:")
    print(f"   2025 Best L/D Ratio: {optimal_ld_2025:.4f} (Peak downforce era)")
    print(f"   2026 Best L/D Ratio: {optimal_ld_2026:.4f} (New regulation era)")
    print(f"   Change: {l_d_change:+.4f} ({l_d_pct:+.2f}%)")
    
    print(f"\n📈 KEY INSIGHTS:")
    print(f"   ✓ 2026 regulations REDUCE downforce as mandated (~30% less)")
    print(f"   ✓ Drag significantly reduced through active aero systems")
    print(f"   ✓ Shift from pure ground effect to active aero blend")
    print(f"   ✓ Cars become lighter and more nimble")
    print(f"   ✓ Better racing through easier overtaking")
    
    print(f"\n⚙️  CRITICAL CHANGES:")
    print(f"   • Downforce Reduction: {downforce_change:.1f}% (FIA target: -30%)")
    print(f"   • Drag Reduction: {drag_change:.1f}% (improved efficiency)")
    print(f"   • Vehicle Weight: Lighter (narrower tyres, smaller components)")
    print(f"   • Primary Downforce Source: Shift to active aero systems")
    
    print(f"\n💡 RECOMMENDATIONS FOR 2026:")
    print(f"   1. Develop active aero systems for drag management (KEY)")
    print(f"   2. Optimize floor/diffuser within new restrictions")
    print(f"   3. Lighter chassis design (narrower tyres, smaller sidepods)")
    print(f"   4. Active suspension for ground effect balance")
    print(f"   5. Focus on mechanical grip to compensate for aero loss")
    
    return {
        '2025': {
            'simulator': simulator_2025,
            'analyzer': analyzer_2025,
            'results': results_2025,
            'optimal': optimal_2025,
            'optimal_ld': optimal_ld_2025
        },
        '2026': {
            'simulator': simulator_2026,
            'analyzer': analyzer_2026,
            'results': results_2026,
            'optimal': optimal_2026,
            'optimal_ld': optimal_ld_2026
        }
    }

if __name__ == "__main__":
    comparison_results = main_comparison()