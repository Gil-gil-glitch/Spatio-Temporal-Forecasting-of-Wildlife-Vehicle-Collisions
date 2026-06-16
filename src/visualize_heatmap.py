"""
Filename: visualize_risk_heatmap.py
Description: Standalone visualization engine for the WVC repository. 
Generates an intuitive Spatio-Temporal risk intensity heatmap along the 
I-40 transportation asset by combining wildlife presence and model predictions.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generate_spatiotemporal_heatmap(telemetry_matrix, save_path="outputs/wvc_risk_heatmap.png"):
    """
    Accepts the 365-day x 96-segment biological matrix and plots a continuous 
    intensity heatmap showing exactly when and where migration risk peaks.
    """
    print("[*] Generating Spatio-Temporal Collision Risk Heatmap...")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Setup the plot configuration
    plt.figure(figsize=(16, 8))
    
    # Transpose the matrix so segments are on the Y-axis (spatial corridor) 
    # and dates are on the X-axis (chronological timeline)
    # This mimics a physical highway driving left-to-right or top-to-bottom
    plot_data = telemetry_matrix.T
    
    # Generate the heatmap
    ax = sns.heatmap(
        plot_data, 
        cmap="YlOrRd",  # Yellow to Orange to deep Red for intuitive risk scaling
        cbar_kws={'label': 'Aggregated Biological Presence Index (Risk Level)'},
        xticklabels=30,  # Show date markers roughly once a month
        yticklabels=5    # Show milestone segment numbers every 25 miles
    )
    
    plt.title('Interstate 40: Spatio-Temporal Wildlife-Vehicle Collision (WVC) Core Risk Topology', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Chronological Timeline (Days across Observation Period)', fontsize=12)
    plt.ylabel('Highway Distance Asset (5-Mile Binned Spatial Segments)', fontsize=12)
    
    # Refine layout aesthetics
    plt.tight_layout()
    
    # Save the output visualization to your repository folder
    plt.savefig(save_path, dpi=300)
    print(f"[+] Intuitive visual heatmap successfully exported to: {save_path}")
    plt.show()

# Example execution template block
if __name__ == "__main__":
    # In production, you would load your saved matrices or call this from your main pipeline:
    # Example:
    # my_matrix = pd.read_csv("data/processed/real_telemetry_matrix.csv", index_col=0, parse_dates=True)
    # generate_spatiotemporal_heatmap(my_matrix)
    print("[!] Script ready. Pass your engineered 'real_telemetry_matrix' to execute.")