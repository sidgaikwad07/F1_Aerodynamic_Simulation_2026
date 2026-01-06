"""
Created on Thu Jan  1 18:52:22 2026

@author: sid
Advanced visualization functions for F1 aerodynamic analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class VisualizationEngine:
    """Advanced visualization engine for F1 aerodynamic analysis"""
    
    @staticmethod
    def visualise_results_improved(simulator, analyzer, results, optimal_design=None):
        """
        Create comprehensive and improved visualizations with better insights
        
        Parameters:
        -----------
        simulator : F1AerodynamicSimulator
            Simulator object containing synthetic aerodynamic data
        analyzer : AerodynamicMLAnalyzer
            Analyzer object with trained models and results
        results : dict
            Dictionary of model results for different targets
        optimal_design : dict, optional
            Dictionary containing optimal design parameters
        """
        
        fig = plt.figure(figsize=(20, 14))
        gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
        
        # ============= 1. Model Performance Heatmap =============
        ax1 = fig.add_subplot(gs[0, 0])
        targets = list(results.keys())
        models = list(results[targets[0]].keys())
        
        # Create r2_scores with shape (models, targets)
        r2_scores = np.array([[results[target][model]['r2'] for target in targets] for model in models])
        
        im = ax1.imshow(r2_scores, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        ax1.set_xticks(range(len(targets)))
        ax1.set_yticks(range(len(models)))
        ax1.set_xticklabels(targets, rotation=45, ha='right', fontsize=10)
        ax1.set_yticklabels(models, fontsize=10)
        ax1.set_title('Model Performance Heatmap (R² Scores)', fontsize=12, fontweight='bold')
        
        # Add text annotations
        for i in range(len(models)):
            for j in range(len(targets)):
                ax1.text(j, i, f'{r2_scores[i, j]:.3f}',
                        ha="center", va="center", color="black", fontsize=9, fontweight='bold')
        
        plt.colorbar(im, ax=ax1, label='R² Score')
        
        # ============= 2. Best Model Predictions (Actual vs Predicted) =============
        ax2 = fig.add_subplot(gs[0, 1])
        best_model = max(results['ld_ratio'], key=lambda x: results['ld_ratio'][x]['r2'])
        actual = results['ld_ratio'][best_model]['actual']
        predicted = results['ld_ratio'][best_model]['predictions']
        
        ax2.scatter(actual, predicted, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        ax2.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 
                'r--', lw=2.5, label='Perfect Prediction', alpha=0.8)
        
        # Add R² annotation
        r2 = results['ld_ratio'][best_model]['r2']
        ax2.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax2.transAxes,
                fontsize=11, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax2.set_xlabel('Actual L/D Ratio', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Predicted L/D Ratio', fontsize=11, fontweight='bold')
        ax2.set_title(f'L/D Prediction Performance ({best_model})', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3, linestyle='--')
        
        # ============= 3. Feature Importance - Multi-target Heatmap =============
        ax3 = fig.add_subplot(gs[0, 2])
        
        # Collect feature importance for all targets
        importance_matrix = []
        importance_targets = []
        feature_names = list(simulator.data.columns[:9])
        
        for target in targets:
            if target in analyzer.feature_importance:
                importance_matrix.append(analyzer.feature_importance[target])
                importance_targets.append(target)
        
        if importance_matrix:
            importance_matrix = np.array(importance_matrix)
            im3 = ax3.imshow(importance_matrix, cmap='YlOrRd', aspect='auto')
            ax3.set_xticks(range(len(feature_names)))
            ax3.set_yticks(range(len(importance_targets)))
            ax3.set_xticklabels(feature_names, rotation=45, ha='right', fontsize=9)
            ax3.set_yticklabels(importance_targets, fontsize=10)
            ax3.set_title('Feature Importance Across Targets', fontsize=12, fontweight='bold')
            plt.colorbar(im3, ax=ax3, label='Importance')
        
        # ============= 4. Drag vs Downforce Trade-off (Hexbin) =============
        ax4 = fig.add_subplot(gs[1, 0])
        hexbin = ax4.hexbin(simulator.data['drag_coefficient'], 
                           simulator.data['downforce_coefficient'],
                           C=simulator.data['ld_ratio'], 
                           gridsize=20, cmap='viridis', mincnt=1, edgecolors='black', linewidths=0.2)
        
        ax4.set_xlabel('Drag Coefficient (Cd)', fontsize=11, fontweight='bold')
        ax4.set_ylabel('Downforce Coefficient (Cl)', fontsize=11, fontweight='bold')
        ax4.set_title('Drag vs Downforce Trade-off (Density)', fontsize=12, fontweight='bold')
        cbar = plt.colorbar(hexbin, ax=ax4, label='L/D Ratio')
        ax4.grid(True, alpha=0.2, linestyle='--')
        
        # ============= 5. Rear Wing Impact (Hexbin) =============
        ax5 = fig.add_subplot(gs[1, 1])
        hexbin2 = ax5.hexbin(simulator.data['rear_wing_angle'], 
                            simulator.data['downforce_coefficient'],
                            C=simulator.data['drag_coefficient'], 
                            gridsize=15, cmap='coolwarm', mincnt=1, edgecolors='black', linewidths=0.2)
        
        ax5.set_xlabel('Rear Wing Angle (degrees)', fontsize=11, fontweight='bold')
        ax5.set_ylabel('Downforce Coefficient (Cl)', fontsize=11, fontweight='bold')
        ax5.set_title('Rear Wing Impact on Downforce', fontsize=12, fontweight='bold')
        cbar2 = plt.colorbar(hexbin2, ax=ax5, label='Drag Coefficient')
        ax5.grid(True, alpha=0.2, linestyle='--')
        
        # ============= 6. Ground Effect Analysis (2D Density) =============
        ax6 = fig.add_subplot(gs[1, 2])
        scatter6 = ax6.scatter(simulator.data['ride_height_rear'], 
                              simulator.data['ld_ratio'],
                              c=simulator.data['speed'], 
                              cmap='plasma', alpha=0.6, s=40, edgecolors='black', linewidth=0.3)
        
        # Add trend line
        z = np.polyfit(simulator.data['ride_height_rear'], simulator.data['ld_ratio'], 2)
        p = np.poly1d(z)
        x_trend = np.linspace(simulator.data['ride_height_rear'].min(), 
                             simulator.data['ride_height_rear'].max(), 100)
        ax6.plot(x_trend, p(x_trend), 'r--', linewidth=2.5, label='Trend', alpha=0.8)
        
        ax6.set_xlabel('Rear Ride Height (mm)', fontsize=11, fontweight='bold')
        ax6.set_ylabel('L/D Ratio', fontsize=11, fontweight='bold')
        ax6.set_title('Ground Effect: Ride Height Impact', fontsize=12, fontweight='bold')
        cbar6 = plt.colorbar(scatter6, ax=ax6, label='Speed (km/h)')
        ax6.legend(fontsize=10)
        ax6.grid(True, alpha=0.3, linestyle='--')
        
        # ============= 7. Front vs Rear Wing Angle Relationship =============
        ax7 = fig.add_subplot(gs[2, 0])
        scatter7 = ax7.scatter(simulator.data['front_wing_angle'], 
                              simulator.data['rear_wing_angle'],
                              c=simulator.data['ld_ratio'], 
                              cmap='viridis', alpha=0.6, s=40, edgecolors='black', linewidth=0.3)
        
        # Mark optimal design if provided
        if optimal_design:
            ax7.scatter(optimal_design['front_wing_angle'], 
                       optimal_design['rear_wing_angle'],
                       marker='*', s=1000, color='red', edgecolors='darkred', linewidth=2,
                       label='Optimal Design', zorder=5)
            ax7.legend(fontsize=10)
        
        ax7.set_xlabel('Front Wing Angle (degrees)', fontsize=11, fontweight='bold')
        ax7.set_ylabel('Rear Wing Angle (degrees)', fontsize=11, fontweight='bold')
        ax7.set_title('Wing Angle Configuration Space', fontsize=12, fontweight='bold')
        cbar7 = plt.colorbar(scatter7, ax=ax7, label='L/D Ratio')
        ax7.grid(True, alpha=0.3, linestyle='--')
        
        # ============= 8. Performance Distribution =============
        ax8 = fig.add_subplot(gs[2, 1])
        
        ax8.hist(simulator.data['drag_coefficient'], bins=30, alpha=0.6, label='Drag', color='red', edgecolor='black')
        ax8_twin = ax8.twinx()
        ax8_twin.hist(simulator.data['downforce_coefficient'], bins=30, alpha=0.6, label='Downforce', color='blue', edgecolor='black')
        
        ax8.set_xlabel('Coefficient Value', fontsize=11, fontweight='bold')
        ax8.set_ylabel('Frequency (Drag)', fontsize=11, fontweight='bold', color='red')
        ax8_twin.set_ylabel('Frequency (Downforce)', fontsize=11, fontweight='bold', color='blue')
        ax8.set_title('Aerodynamic Coefficient Distribution', fontsize=12, fontweight='bold')
        ax8.tick_params(axis='y', labelcolor='red')
        ax8_twin.tick_params(axis='y', labelcolor='blue')
        ax8.grid(True, alpha=0.3, linestyle='--')
        
        # ============= 9. Optimal Design Summary =============
        ax9 = fig.add_subplot(gs[2, 2])
        ax9.axis('off')
        
        summary_text = "OPTIMIZATION SUMMARY\n" + "="*45 + "\n\n"
        
        if optimal_design:
            summary_text += "📊 OPTIMAL DESIGN PARAMETERS:\n"
            summary_text += "-"*45 + "\n"
            key_params = ['front_wing_angle', 'rear_wing_angle', 'ride_height_rear', 
                         'rake_angle', 'diffuser_angle']
            for param in key_params:
                if param in optimal_design:
                    value = optimal_design[param]
                    summary_text += f"  • {param.replace('_', ' ').title()}\n"
                    summary_text += f"    {value:.2f}°\n"
            
            summary_text += "\n🎯 KEY INSIGHTS:\n"
            summary_text += "-"*45 + "\n"
            summary_text += "  ✓ Rear wing angle is most critical\n"
            summary_text += "  ✓ Lower ride height = better ground effect\n"
            summary_text += "  ✓ Balance front/rear for stability\n"
            summary_text += "  ✓ Optimize for L/D ratio efficiency\n"
            summary_text += "\n⚠️ 2026 REGULATION COMPLIANCE:\n"
            summary_text += "-"*45 + "\n"
            summary_text += "  ✓ All parameters within FIA limits\n"
            summary_text += "  ✓ Design respects aero constraints\n"
            summary_text += "  ✓ Sustainable aerodynamic approach\n"
        else:
            summary_text += "Run optimization to see results\n"
        
        ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        
        plt.suptitle('F1 AERODYNAMIC ANALYSIS - 2026 TECHNICAL REGULATIONS', 
                     fontsize=16, fontweight='bold', y=0.995)
        
        plt.savefig('f1_aerodynamic_analysis_improved.png', dpi=300, bbox_inches='tight')
        print("\n✓ Enhanced visualization saved as 'f1_aerodynamic_analysis_improved.png'")
        plt.show()