import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib.image as mpimg
import textwrap


from sandbox.constants import LAYOUTS_IMAGES_DIR, QUESTIONNAIRES_DIR, AgentType, TrainedOnLayouts, LearnerType


def plot_best_type_layouts(agent_types, questionnaire_file_name, layouts_prefix, trained_on_number_of_layouts, learner_types):
    df = pd.read_csv(f'{QUESTIONNAIRES_DIR}/{questionnaire_file_name}/{questionnaire_file_name}.csv')
    df = df[df['Which layout? (write the layout name as it exactly appears on the repository)'].str.contains(layouts_prefix, na=False)]
    df = df[df['Reward'] != 'N/A']
    df = df[df['Trained on ... Layout(s)'].str.contains(trained_on_number_of_layouts, na=False)]
    df['Reward'] = pd.to_numeric(df['Reward'], errors='coerce')
    df = df.dropna(subset=['Reward'])
    df = df[df['LearnerType'].isin(learner_types)]

    # show unique maxs w.r.t to agent types
    exclude_agent_types = [agent for agent in AgentType.ALL if agent not in agent_types]
    df_exclude_agents_max_rewards = df[df['Agent Type'].isin(exclude_agent_types)].groupby('Which layout? (write the layout name as it exactly appears on the repository)')['Reward'].max()
    df_max_rewards = df[df['Agent Type'].isin(agent_types)].groupby('Which layout? (write the layout name as it exactly appears on the repository)')['Reward'].max()
    df_exclude_agents_max_rewards = df_exclude_agents_max_rewards.reindex(df_max_rewards.index, fill_value=float('-inf'))
    df_max_rewards_filtered = df_max_rewards[df_max_rewards >= df_exclude_agents_max_rewards]
    best_agent_type_layouts = df_max_rewards_filtered.index
    df_best_agent_type = df[df['Which layout? (write the layout name as it exactly appears on the repository)'].isin(best_agent_type_layouts)]

    unique_layouts = best_agent_type_layouts.unique()

    fig, axes = plt.subplots(nrows=len(unique_layouts), ncols=2, figsize=(27, 5 * len(unique_layouts)))

    if len(unique_layouts) == 1:
        axes = [axes]

    for i, layout in enumerate(unique_layouts):
        layout_df = df_best_agent_type[df_best_agent_type['Which layout? (write the layout name as it exactly appears on the repository)'] == layout]
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

        for p, note in zip(axes[i, 0].patches, layout_df['Notes']):
            # reward
            axes[i, 0].annotate(format(p.get_height(), '.2f'),
                            (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha = 'center', va = 'center',
                            xytext = (0, 9),
                            textcoords = 'offset points')
            # notes
            if not isinstance(note, str):
                note = ""
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
    plt.tight_layout()

    if layouts_prefix == '':
        layouts_prefix = 'all'
    if trained_on_number_of_layouts == '':
        trained_on_number_of_layouts = 'OneOrMultiple'
    # replace / with _ in agent types
    agent_types = '_'.join(agent_types).replace('/', '+')

    plt.savefig(f'{QUESTIONNAIRES_DIR}/{questionnaire_file_name}/best_{agent_types}_using_LT_{learner_types}_in_{layouts_prefix}_layouts_trained_on_{trained_on_number_of_layouts}_layouts.png', dpi=100)


if __name__ == "__main__":
    questionnaire_file_name = '2'
    layouts_prefix = ''

    # agent_types = [AgentType.n_1_sp_new_cur, AgentType.n_1_sp_ran, AgentType.n_1_sp_w_cur]
    agent_types = [AgentType.n_1_sp_w_cur]
    learner_types = [LearnerType.originaler]
    trained_on_number_of_layouts = TrainedOnLayouts.multiple # TrainedOnLayouts.multiple, TrainedOnLayouts.one

    plot_best_type_layouts(agent_types=agent_types,
                           questionnaire_file_name=questionnaire_file_name,
                           layouts_prefix=layouts_prefix,
                           trained_on_number_of_layouts=trained_on_number_of_layouts,\
                           learner_types=learner_types
                           )
