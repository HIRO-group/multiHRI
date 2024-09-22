import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.image as mpimg
import os
import textwrap

from sandbox.constants import LAYOUTS_IMAGES_DIR, QUESTIONNAIRES_DIR, AgentType, TrainedOnLayouts, LearnerType

def plot_agent_comparison(agent_type_A, agent_type_B, questionnaire_file_name, layouts_prefix, trained_on_number_of_layouts, learner_types):
    df = pd.read_csv(f'{QUESTIONNAIRES_DIR}/{questionnaire_file_name}/{questionnaire_file_name}.csv')
    df = df[df['Which layout? (write the layout name as it exactly appears on the repository)'].str.contains(layouts_prefix, na=False)]

    # #List of layout names to exclude
    # exclude_layouts = [
    # #     "3_chefs_counter_circuit",
    # #     "3_chefs_coordination_ring",
    # #     "3_chefs_small_kitchen",
    # #     "3_chefs_small_kitchen_two_resources",
    # #     "3_chefs_4P_4O_4D_4S",
    #     '3_chefs_forced_coordination_3OP2S1D', 
    #     '3_chefs_forced_coordination',
    # #     '3_chefs_storage_room'
    # ]
    # df = df[~df['Which layout? (write the layout name as it exactly appears on the repository)'].isin(exclude_layouts)]

    df = df[df['Reward'] != 'N/A']
    df = df[df['Trained on ... Layout(s)'].str.contains(trained_on_number_of_layouts, na=False)]
    df['Reward'] = pd.to_numeric(df['Reward'], errors='coerce')
    df = df.dropna(subset=['Reward'])
    
    df_A = df[(df['Agent Type'] == agent_type_A) & (df['LearnerType'].isin(learner_types))]
    df_B = df[(df['Agent Type'] == agent_type_B) & (df['LearnerType'].isin(learner_types))]

    merged_df = pd.merge(df_A, df_B, on='Which layout? (write the layout name as it exactly appears on the repository)', suffixes=('_A', '_B'))
    merged_df['Reward_Difference'] = merged_df['Reward_A'] - merged_df['Reward_B']

    layouts_to_plot = merged_df[merged_df['Reward_Difference'] > 0]['Which layout? (write the layout name as it exactly appears on the repository)'].unique()

    if len(layouts_to_plot) == 0:
        print("No layouts found where agent_type_A has a higher reward than agent_type_B.")
        return

    fig, axes = plt.subplots(nrows=len(layouts_to_plot), ncols=2, figsize=(27, 5 * len(layouts_to_plot)))

    if len(layouts_to_plot) == 1:
        axes = [axes]

    for i, layout in enumerate(layouts_to_plot):
        layout_df = df[df['Which layout? (write the layout name as it exactly appears on the repository)'] == layout]
        layout_df = layout_df[layout_df['LearnerType'].isin(learner_types)]
        layout_df = layout_df.sort_values(by='Reward', ascending=False)
        
        layout_df['Label'] = layout_df.apply(
            lambda row: f"{row['Agent Type']}\n{row['LearnerType']}\n Trained on {row['Trained on ... Layout(s)']} layout(s)",
            axis=1
        )
        
        sns.barplot(x='Label', y='Reward', data=layout_df, ax=axes[i, 0], palette='viridis')
        axes[i, 0].set_title(f"Layout: {layout} - Sorted by Reward")
        axes[i, 0].set_ylabel("Reward")
        axes[i, 0].tick_params(axis='x', rotation=45)
        max_reward = layout_df['Reward'].max()
        axes[i, 0].set_ylim(0, max_reward * 1.2)

        for patch, (agent_type, learner_type) in zip(axes[i, 0].patches, layout_df[['Agent Type', 'LearnerType']].values):
            if agent_type == agent_type_A and learner_type in learner_types:
                patch.set_facecolor('yellowgreen')
            elif agent_type == agent_type_B and learner_type in learner_types:
                patch.set_facecolor('tomato')
            else:
                patch.set_facecolor('grey')

        for p, note in zip(axes[i, 0].patches, layout_df['Notes']):
            axes[i, 0].annotate(format(p.get_height(), '.2f'), 
                                (p.get_x() + p.get_width() / 2., p.get_height()), 
                                ha='center', va='center', 
                                xytext=(0, 9), textcoords='offset points')
            
            if isinstance(note, str):
                wrapped_note = "\n".join(textwrap.wrap(note, width=15))
                lines = wrapped_note.split("\n")
                n_lines = len(lines)
                line_height = p.get_height() / (n_lines + 1)

                for j, line in enumerate(lines):
                    y_pos = p.get_height() - (j + 1) * line_height
                    axes[i, 0].text(p.get_x() + p.get_width() / 2., y_pos, 
                                    line, ha='center', va='center', 
                                    fontsize=10, color='white')

        layout_image_path = os.path.join(LAYOUTS_IMAGES_DIR, f"{layout}/-1.png")
        if os.path.exists(layout_image_path):
            img = mpimg.imread(layout_image_path)
            axes[i, 1].imshow(img)
            axes[i, 1].axis('off')
        else:
            axes[i, 1].text(0.5, 0.5, "Image not found", ha='center', va='center', fontsize=12)
            axes[i, 1].axis('off')

    if layouts_prefix == '': layouts_prefix = 'all'
    if trained_on_number_of_layouts == '': trained_on_number_of_layouts = 'OneOrMultiple'

    plt.tight_layout()
    agent_type_A = agent_type_A.replace('/', '_')
    agent_type_B = agent_type_B.replace('/', '_')


    plt.savefig(f'{QUESTIONNAIRES_DIR}/{questionnaire_file_name}/compare_{agent_type_A}_>_{agent_type_B}_highlighted.png', dpi=100)

if __name__ == "__main__":
    questionnaire_file_name = '2'
    layouts_prefix = ''
    
    agent_type_A = AgentType.n_1_sp_new_cur
    agent_type_B = AgentType.n_1_sp_w_cur
    learner_types = [LearnerType.originaler]
    trained_on_number_of_layouts = TrainedOnLayouts.multiple 

    plot_agent_comparison(agent_type_A=agent_type_A,
                          agent_type_B=agent_type_B,
                          questionnaire_file_name=questionnaire_file_name,
                          layouts_prefix=layouts_prefix,
                          trained_on_number_of_layouts=trained_on_number_of_layouts,
                          learner_types=learner_types
                          )
