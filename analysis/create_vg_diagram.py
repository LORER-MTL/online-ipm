"""
Create a diagram showing precisely what V_g^{ineq} measures
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Left panel: Time series showing g_t changes
ax1.set_title(r'Definition: $V_g^{\mathrm{ineq}} = \sum_{t=1}^T \|g_t - g_{t-1}\|$', 
              fontsize=14, fontweight='bold', pad=15)
ax1.set_xlabel('Time t', fontsize=12)
ax1.set_ylabel(r'$\|g_t\|$ (magnitude)', fontsize=12)

# Example g_t trajectory
np.random.seed(42)
T = 10
g_norms = np.zeros(T+1)
g_norms[0] = 5.0
for t in range(1, T+1):
    g_norms[t] = g_norms[t-1] + np.random.randn() * 0.5 + 0.1

times = np.arange(T+1)
ax1.plot(times, g_norms, 'o-', color='#2E86AB', linewidth=2.5, markersize=8, 
         label=r'$\|g_t\|$', zorder=3)

# Annotate differences
for t in range(1, 6):
    mid_x = (times[t-1] + times[t]) / 2
    mid_y = (g_norms[t-1] + g_norms[t]) / 2
    delta = abs(g_norms[t] - g_norms[t-1])
    ax1.annotate('', xy=(times[t], g_norms[t]), xytext=(times[t-1], g_norms[t-1]),
                arrowprops=dict(arrowstyle='<->', color='red', lw=2))
    ax1.text(mid_x, mid_y + 0.3, f'Δ={delta:.2f}', fontsize=9, 
            ha='center', color='red', fontweight='bold')

ax1.grid(True, alpha=0.3, linestyle='--')
ax1.legend(fontsize=11, loc='upper left')
ax1.set_xlim(-0.5, 10.5)
ax1.text(5, g_norms.max() + 0.8, 
        r'$V_g^{\mathrm{ineq}}$ = sum of red arrows',
        fontsize=11, ha='center', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

# Right panel: Component breakdown
ax2.set_title('What Changes What in the Augmented Problem', 
              fontsize=14, fontweight='bold', pad=15)
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)
ax2.axis('off')

# Box for environment changes
env_box = patches.FancyBboxPatch((0.5, 7), 4, 2, boxstyle="round,pad=0.1",
                                  edgecolor='#A23B72', facecolor='#F18F01', alpha=0.3, linewidth=2.5)
ax2.add_patch(env_box)
ax2.text(2.5, 8, 'Environment Changes', fontsize=11, ha='center', fontweight='bold')
ax2.text(2.5, 7.4, r'$g_t$ varies over time', fontsize=10, ha='center', style='italic')

# Box for solution changes
sol_box = patches.FancyBboxPatch((5.5, 7), 4, 2, boxstyle="round,pad=0.1",
                                 edgecolor='#006BA6', facecolor='#0496FF', alpha=0.3, linewidth=2.5)
ax2.add_patch(sol_box)
ax2.text(7.5, 8, 'Solution Changes', fontsize=11, ha='center', fontweight='bold')
ax2.text(7.5, 7.4, r'$x_t^*$ adjusts optimally', fontsize=10, ha='center', style='italic')

# Arrow showing coupling
ax2.annotate('', xy=(7.5, 6.8), xytext=(2.5, 6.8),
            arrowprops=dict(arrowstyle='->', lw=3, color='black'))
ax2.text(5, 6.3, 'affects', fontsize=10, ha='center', style='italic')

# Box for slack changes
slack_box = patches.FancyBboxPatch((2, 4), 6, 1.8, boxstyle="round,pad=0.1",
                                   edgecolor='#540D6E', facecolor='#EE4266', alpha=0.3, linewidth=2.5)
ax2.add_patch(slack_box)
ax2.text(5, 5.3, r'Slack Variable Changes: $s_t^* = g_t - Fx_t^*$', 
        fontsize=11, ha='center', fontweight='bold')
ax2.text(5, 4.6, r'$\|s_t^* - s_{t-1}^*\| \leq \|g_t - g_{t-1}\| + \|F\|\cdot\|x_t^* - x_{t-1}^*\|$',
        fontsize=10, ha='center', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Arrows from components to slack
ax2.annotate('', xy=(3.5, 5.8), xytext=(2.5, 6.8),
            arrowprops=dict(arrowstyle='->', lw=2.5, color='#A23B72'))
ax2.text(2.8, 6.2, r'$V_g^{\mathrm{ineq}}$', fontsize=10, color='#A23B72', fontweight='bold')

ax2.annotate('', xy=(6.5, 5.8), xytext=(7.5, 6.8),
            arrowprops=dict(arrowstyle='->', lw=2.5, color='#006BA6'))
ax2.text(7.2, 6.2, r'$\|F\| V_T^x$', fontsize=10, color='#006BA6', fontweight='bold')

# Final bound box
bound_box = patches.FancyBboxPatch((1, 1.5), 8, 2, boxstyle="round,pad=0.15",
                                   edgecolor='darkgreen', facecolor='lightgreen', alpha=0.4, linewidth=3)
ax2.add_patch(bound_box)
ax2.text(5, 2.9, 'Total Slack Variation:', fontsize=11, ha='center', fontweight='bold')
ax2.text(5, 2.3, r'$V_T^s \leq V_g^{\mathrm{ineq}} + \|F\| \cdot V_T^x$',
        fontsize=12, ha='center', 
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='darkgreen', linewidth=2))

# Final regret bound
ax2.text(5, 0.8, r'$\Rightarrow$ Regret: $R_d(T) \leq \mathrm{const} + \|c\|[(1+\|F\|)V_T^x + V_g^{\mathrm{ineq}}]$',
        fontsize=10, ha='center', style='italic',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

# Add legend for clarity
legend_text = [
    r'$V_g^{\mathrm{ineq}}$: Inequality RHS variation (environment)',
    r'$V_T^x$: Original variable path variation (solution)',
    r'$V_T^s$: Slack variable path variation (derived)'
]
for i, text in enumerate(legend_text):
    ax2.text(0.5, 0.3 - i*0.15, text, fontsize=9, transform=ax2.transAxes,
            verticalalignment='top')

plt.tight_layout()
plt.savefig('analysis/vg_ineq_definition.png', dpi=150, bbox_inches='tight')
print("Created: analysis/vg_ineq_definition.png")
