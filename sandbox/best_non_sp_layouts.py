import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib.image as mpimg

layout_image_dir = 'data/layouts'
df = pd.read_csv('data/questionnaire/1.csv')

df_3_chefs = df[df['Which layout? (write the layout name as it exactly appears on the repository)'].str.contains('', na=False)]

df_3_chefs = df_3_chefs[df_3_chefs['Reward'] != 'N/A']
df_3_chefs['Reward'] = pd.to_numeric(df_3_chefs['Reward'], errors='coerce')

df_3_chefs = df_3_chefs.dropna(subset=['Reward'])

# Group by layout and find the highest reward for non-Self play agents
non_sp_max_rewards = df_3_chefs[df_3_chefs['Agent Type'] != 'Self play'].groupby('Which layout? (write the layout name as it exactly appears on the repository)')['Reward'].max()

# Filter layouts where non-Self play agents have the highest reward
best_non_sp_layouts = non_sp_max_rewards[non_sp_max_rewards == df_3_chefs.groupby('Which layout? (write the layout name as it exactly appears on the repository)')['Reward'].max()].index

# Filter the dataframe to include only the best non-Self play layouts
df_best_non_sp = df_3_chefs[df_3_chefs['Which layout? (write the layout name as it exactly appears on the repository)'].isin(best_non_sp_layouts)]

unique_layouts = best_non_sp_layouts.unique()

fig, axes = plt.subplots(nrows=len(unique_layouts), ncols=2, figsize=(27, 5 * len(unique_layouts)))

if len(unique_layouts) == 1:
    axes = [axes]

for i, layout in enumerate(unique_layouts):
    layout_df = df_best_non_sp[df_best_non_sp['Which layout? (write the layout name as it exactly appears on the repository)'] == layout]
    
    layout_df = layout_df.sort_values(by='Reward', ascending=False)
    layout_df['Label'] = layout_df.apply(
        lambda row: f"{row['Agent Type']}\n{row['LearnerType']}\n Trained on {row['Trained on ... Layout(s)']} layout(s)",
        axis=1
    )
    sns.barplot(x='Label', y='Reward', data=layout_df, ax=axes[i, 0], palette='viridis')
    
    # Find the best non-Self play agent type
    best_agent = layout_df[layout_df['Agent Type'] != 'Self play'].iloc[0]['Agent Type']
    
    axes[i, 0].set_title(f"Layout: {layout} - Sorted by Reward (Best: {best_agent})")
    axes[i, 0].set_ylabel("Reward")
    axes[i, 0].tick_params(axis='x', rotation=45)

    max_reward = layout_df['Reward'].max()

    axes[i, 0].set_ylim(0, max_reward * 1.2)

    for p in axes[i, 0].patches:
        axes[i, 0].annotate(format(p.get_height(), '.2f'), 
                         (p.get_x() + p.get_width() / 2., p.get_height()), 
                         ha = 'center', va = 'center', 
                         xytext = (0, 9), 
                         textcoords = 'offset points')

    layout_image_path = os.path.join(layout_image_dir, f"{layout}/-1.png")
    if os.path.exists(layout_image_path):
        img = mpimg.imread(layout_image_path)
        axes[i, 1].imshow(img)
        axes[i, 1].axis('off')
    else:
        axes[i, 1].text(0.5, 0.5, "Image not found", ha='center', va='center', fontsize=12)
        axes[i, 1].axis('off')

plt.tight_layout()
plt.savefig('data/questionnaire/best_non_sp_layouts.png', dpi=100)