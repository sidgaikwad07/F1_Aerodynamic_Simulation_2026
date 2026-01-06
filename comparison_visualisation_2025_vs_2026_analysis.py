"""
Created on Mon Jan  5 19:50:30 2026

@author: sid
F1 Aerodynamic Comparison Visualisation Module
2025 v 2026 Technical Regulations

Comparative analysis of aerodynamic regulations and performance changes
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class ComparisonVisualizationEngine:
    """Advanced visualization engine for comparing F1 aerodynamics across seasons"""
    
    @staticmethod
    def visualise_2025_vs_2026(simulator_2025, analyzer_2025, results_2025, optimal_2025,
                               simulator_2026, analyzer_2026, results_2026, optimal_2026):
        """
        Create comprehensive comparison visualizations between 2025 and 2026 regulations
        
        Parameters:
        -----------
        simulator_2025 : F1AerodynamicSimulator
            2025 simulator object
        analyzer_2025 : AerodynamicMLAnalyzer
            2025 analyzer with trained models
        results_2025 : dict
            2025 model results
        optimal_2025 : dict
            2025 optimal design parameters
        simulator_2026 : F1AerodynamicSimulator
            2026 simulator object
        analyzer_2026 : AerodynamicMLAnalyzer
            2026 analyzer with trained models
        results_2026 : dict
            2026 model results
        optimal_2026 : dict
            2026 optimal design parameters
        """
        
        fig = plt.figure(figsize=(22, 16))
        gs = fig.add_gridspec(4, 3, hspace=0.4, wspace=0.32)
        
        # ============= 1. Model Performance Comparison =============
        ax1 = fig.add_subplot(gs[0, :2])
        
        targets = list(results_2025.keys())
        models = list(results_2025[targets[0]].keys())
        
        # Get R² scores for both years
        r2_2025 = np.array([[results_2025[target][model]['r2'] for target in targets] for model in models])
        r2_2026 = np.array([[results_2026[target][model]['r2'] for target in targets] for model in models])
        
        x = np.arange(len(models))
        width = 0.15
        
        for i, target in enumerate(targets):
            offset = (i - len(targets)/2) * width
            vals_2025 = [results_2025[target][model]['r2'] for model in models]
            vals_2026 = [results_2026[target][model]['r2'] for model in models]
            
            ax1.bar(x + offset - width/2, vals_2025, width/2, label=f'{target} (2025)', alpha=0.8)
            ax1.bar(x + offset + width/2, vals_2026, width/2, label=f'{target} (2026)', alpha=0.8)
        
        ax1.set_xlabel('Model Type', fontsize=12, fontweight='bold')
        ax1.set_ylabel('R² Score', fontsize=12, fontweight='bold')
        ax1.set_title('Model Performance: 2025 vs 2026 Regulations', fontsize=13, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(models, fontsize=11)
        ax1.legend(fontsize=9, ncol=4, loc='upper left')
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.set_ylim([0, 1.1])
        
        # ============= 2. Regulation Complexity Comparison =============
        ax2 = fig.add_subplot(gs[0, 2])
        
        regulations_2025 = {
            'Front Wing\nComplexity': 0.65,
            'Rear Wing\nComplexity': 0.70,
            'Floor/Diffuser\nComplexity': 0.75,
            'DRS System': 0.60
        }
        
        regulations_2026 = {
            'Front Wing\nComplexity': 0.78,
            'Rear Wing\nComplexity': 0.82,
            'Floor/Diffuser\nComplexity': 0.88,
            'Active Aero': 0.95
        }
        
        reg_names = list(regulations_2025.keys())
        vals_2025 = list(regulations_2025.values())
        vals_2026 = list(regulations_2026.values())
        
        x_reg = np.arange(len(reg_names))
        width_reg = 0.35
        
        ax2.bar(x_reg - width_reg/2, vals_2025, width_reg, label='2025', alpha=0.8, color='skyblue')
        ax2.bar(x_reg + width_reg/2, vals_2026, width_reg, label='2026', alpha=0.8, color='coral')
        
        ax2.set_ylabel('Complexity Index', fontsize=11, fontweight='bold')
        ax2.set_title('Aerodynamic Regulation Changes', fontsize=12, fontweight='bold')
        ax2.set_xticks(x_reg)
        ax2.set_xticklabels(reg_names, fontsize=9)
        ax2.legend(fontsize=10)
        ax2.set_ylim([0, 1.0])
        ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
        
        # ============= 3. Drag Coefficient Distribution Comparison =============
        ax3 = fig.add_subplot(gs[1, 0])
        
        ax3.hist(simulator_2025.data['drag_coefficient'], bins=25, alpha=0.6, 
                label='2025', color='blue', edgecolor='black', density=True)
        ax3.hist(simulator_2026.data['drag_coefficient'], bins=25, alpha=0.6, 
                label='2026', color='red', edgecolor='black', density=True)
        
        mean_2025_drag = simulator_2025.data['drag_coefficient'].mean()
        mean_2026_drag = simulator_2026.data['drag_coefficient'].mean()
        
        ax3.axvline(mean_2025_drag, color='blue', linestyle='--', linewidth=2.5, label=f'2025 Mean: {mean_2025_drag:.3f}')
        ax3.axvline(mean_2026_drag, color='red', linestyle='--', linewidth=2.5, label=f'2026 Mean: {mean_2026_drag:.3f}')
        
        ax3.set_xlabel('Drag Coefficient (Cd)', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Density', fontsize=11, fontweight='bold')
        ax3.set_title('Drag Coefficient Evolution', fontsize=12, fontweight='bold')
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3, linestyle='--')
        
        # ============= 4. Downforce Coefficient Distribution Comparison =============
        ax4 = fig.add_subplot(gs[1, 1])
        
        ax4.hist(simulator_2025.data['downforce_coefficient'], bins=25, alpha=0.6, 
                label='2025', color='green', edgecolor='black', density=True)
        ax4.hist(simulator_2026.data['downforce_coefficient'], bins=25, alpha=0.6, 
                label='2026', color='orange', edgecolor='black', density=True)
        
        mean_2025_df = simulator_2025.data['downforce_coefficient'].mean()
        mean_2026_df = simulator_2026.data['downforce_coefficient'].mean()
        
        ax4.axvline(mean_2025_df, color='green', linestyle='--', linewidth=2.5, label=f'2025 Mean: {mean_2025_df:.3f}')
        ax4.axvline(mean_2026_df, color='orange', linestyle='--', linewidth=2.5, label=f'2026 Mean: {mean_2026_df:.3f}')
        
        ax4.set_xlabel('Downforce Coefficient (Cl)', fontsize=11, fontweight='bold')
        ax4.set_ylabel('Density', fontsize=11, fontweight='bold')
        ax4.set_title('Downforce Coefficient Evolution', fontsize=12, fontweight='bold')
        ax4.legend(fontsize=9)
        ax4.grid(True, alpha=0.3, linestyle='--')
        
        # ============= 5. L/D Ratio Efficiency Comparison =============
        ax5 = fig.add_subplot(gs[1, 2])
        
        ax5.hist(simulator_2025.data['ld_ratio'], bins=25, alpha=0.6, 
                label='2025', color='purple', edgecolor='black', density=True)
        ax5.hist(simulator_2026.data['ld_ratio'], bins=25, alpha=0.6, 
                label='2026', color='brown', edgecolor='black', density=True)
        
        mean_2025_ld = simulator_2025.data['ld_ratio'].mean()
        mean_2026_ld = simulator_2026.data['ld_ratio'].mean()
        
        ax5.axvline(mean_2025_ld, color='purple', linestyle='--', linewidth=2.5, label=f'2025 Mean: {mean_2025_ld:.3f}')
        ax5.axvline(mean_2026_ld, color='brown', linestyle='--', linewidth=2.5, label=f'2026 Mean: {mean_2026_ld:.3f}')
        
        ax5.set_xlabel('L/D Ratio', fontsize=11, fontweight='bold')
        ax5.set_ylabel('Density', fontsize=11, fontweight='bold')
        ax5.set_title('Aerodynamic Efficiency Evolution', fontsize=12, fontweight='bold')
        ax5.legend(fontsize=9)
        ax5.grid(True, alpha=0.3, linestyle='--')
        
        # ============= 6. Drag vs Downforce Comparison (2025) =============
        ax6 = fig.add_subplot(gs[2, 0])
        
        scatter6 = ax6.scatter(simulator_2025.data['drag_coefficient'], 
                              simulator_2025.data['downforce_coefficient'],
                              c=simulator_2025.data['ld_ratio'], 
                              cmap='viridis', alpha=0.6, s=30, edgecolors='black', linewidth=0.3)
        
        ax6.set_xlabel('Drag Coefficient (Cd)', fontsize=11, fontweight='bold')
        ax6.set_ylabel('Downforce Coefficient (Cl)', fontsize=11, fontweight='bold')
        ax6.set_title('2025: Drag vs Downforce Trade-off', fontsize=12, fontweight='bold')
        cbar6 = plt.colorbar(scatter6, ax=ax6, label='L/D Ratio')
        ax6.grid(True, alpha=0.3, linestyle='--')
        
        # ============= 7. Drag vs Downforce Comparison (2026) =============
        ax7 = fig.add_subplot(gs[2, 1])
        
        scatter7 = ax7.scatter(simulator_2026.data['drag_coefficient'], 
                              simulator_2026.data['downforce_coefficient'],
                              c=simulator_2026.data['ld_ratio'], 
                              cmap='viridis', alpha=0.6, s=30, edgecolors='black', linewidth=0.3)
        
        ax7.set_xlabel('Drag Coefficient (Cd)', fontsize=11, fontweight='bold')
        ax7.set_ylabel('Downforce Coefficient (Cl)', fontsize=11, fontweight='bold')
        ax7.set_title('2026: Drag vs Downforce Trade-off', fontsize=12, fontweight='bold')
        cbar7 = plt.colorbar(scatter7, ax=ax7, label='L/D Ratio')
        ax7.grid(True, alpha=0.3, linestyle='--')
        
        # ============= 8. Performance Metrics Summary =============
        ax8 = fig.add_subplot(gs[2, 2])
        
        metrics_data = {
            'Metric': ['Avg Drag', 'Avg Downforce', 'Avg L/D', 'Max L/D', 'Min Drag'],
            '2025': [
                simulator_2025.data['drag_coefficient'].mean(),
                simulator_2025.data['downforce_coefficient'].mean(),
                simulator_2025.data['ld_ratio'].mean(),
                simulator_2025.data['ld_ratio'].max(),
                simulator_2025.data['drag_coefficient'].min()
            ],
            '2026': [
                simulator_2026.data['drag_coefficient'].mean(),
                simulator_2026.data['downforce_coefficient'].mean(),
                simulator_2026.data['ld_ratio'].mean(),
                simulator_2026.data['ld_ratio'].max(),
                simulator_2026.data['drag_coefficient'].min()
            ]
        }
        
        metrics_df = pd.DataFrame(metrics_data)
        
        x_metrics = np.arange(len(metrics_df))
        width_metrics = 0.35
        
        ax8.bar(x_metrics - width_metrics/2, metrics_df['2025'], width_metrics, label='2025', alpha=0.8)
        ax8.bar(x_metrics + width_metrics/2, metrics_df['2026'], width_metrics, label='2026', alpha=0.8)
        
        ax8.set_ylabel('Value', fontsize=11, fontweight='bold')
        ax8.set_title('Aerodynamic Metrics Comparison', fontsize=12, fontweight='bold')
        ax8.set_xticks(x_metrics)
        ax8.set_xticklabels(metrics_df['Metric'], rotation=45, ha='right', fontsize=9)
        ax8.legend(fontsize=10)
        ax8.grid(True, alpha=0.3, axis='y', linestyle='--')
        
        # ============= 9. Rear Wing Angle Impact Comparison (2025) =============
        ax9 = fig.add_subplot(gs[3, 0])
        
        hexbin9 = ax9.hexbin(simulator_2025.data['rear_wing_angle'], 
                            simulator_2025.data['downforce_coefficient'],
                            C=simulator_2025.data['drag_coefficient'], 
                            gridsize=12, cmap='coolwarm', mincnt=1, edgecolors='black', linewidths=0.2)
        
        ax9.set_xlabel('Rear Wing Angle (°)', fontsize=11, fontweight='bold')
        ax9.set_ylabel('Downforce (Cl)', fontsize=11, fontweight='bold')
        ax9.set_title('2025: Rear Wing Impact', fontsize=12, fontweight='bold')
        cbar9 = plt.colorbar(hexbin9, ax=ax9, label='Drag')
        
        # ============= 10. Rear Wing Angle Impact Comparison (2026) =============
        ax10 = fig.add_subplot(gs[3, 1])
        
        hexbin10 = ax10.hexbin(simulator_2026.data['rear_wing_angle'], 
                             simulator_2026.data['downforce_coefficient'],
                             C=simulator_2026.data['drag_coefficient'], 
                             gridsize=12, cmap='coolwarm', mincnt=1, edgecolors='black', linewidths=0.2)
        
        ax10.set_xlabel('Rear Wing Angle (°)', fontsize=11, fontweight='bold')
        ax10.set_ylabel('Downforce (Cl)', fontsize=11, fontweight='bold')
        ax10.set_title('2026: Rear Wing Impact', fontsize=12, fontweight='bold')
        cbar10 = plt.colorbar(hexbin10, ax=ax10, label='Drag')
        
        # ============= 11. Optimal Design Comparison Summary =============
        ax11 = fig.add_subplot(gs[3, 2])
        ax11.axis('off')
        
        summary_text = "OPTIMAL DESIGN COMPARISON\n" + "="*50 + "\n\n"
        
        summary_text += "2025 OPTIMAL PARAMETERS:\n"
        summary_text += "-"*50 + "\n"
        key_params = ['front_wing_angle', 'rear_wing_angle', 'ride_height_rear', 'rake_angle']
        for param in key_params:
            if param in optimal_2025:
                summary_text += f"  {param.replace('_', ' ').title()}\n"
                summary_text += f"    {optimal_2025[param]:.2f}°\n"
        
        summary_text += "\n2026 OPTIMAL PARAMETERS:\n"
        summary_text += "-"*50 + "\n"
        for param in key_params:
            if param in optimal_2026:
                summary_text += f"  {param.replace('_', ' ').title()}\n"
                summary_text += f"    {optimal_2026[param]:.2f}°\n"
        
        summary_text += "\n⚡ KEY CHANGES:\n"
        summary_text += "-"*50 + "\n"
        summary_text += "  • More active aerodynamics (2026)\n"
        summary_text += "  • Higher downforce efficiency\n"
        summary_text += "  • Stricter drag regulations\n"
        summary_text += "  • Enhanced stability requirements\n"
        
        ax11.text(0.05, 0.95, summary_text, transform=ax11.transAxes, fontsize=9.5,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.4, pad=1))
        
        plt.suptitle('F1 AERODYNAMIC ANALYSIS: 2025 vs 2026 TECHNICAL REGULATIONS COMPARISON', 
                     fontsize=16, fontweight='bold', y=0.995)
        
        plt.savefig('f1_aerodynamic_2025_vs_2026_comparison.png', dpi=300, bbox_inches='tight')
        print("\n✓ Comparison visualization saved as 'f1_aerodynamic_2025_vs_2026_comparison.png'")
        plt.show()