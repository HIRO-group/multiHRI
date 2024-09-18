import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg
import textwrap

from sandbox.constants import LAYOUTS_IMAGES_DIR, QUESTIONNAIRES_DIR, AgentType, TrainedOnLayouts, LearnerType

def summarize_in_a_figure(questionnaire_file_name, layouts_prefix, trained_on_number_of_layouts):
    df = pd.read_csv(f'{QUESTIONNAIRES_DIR}/{questionnaire_file_name}/{questionnaire_file_name}.csv')
    df = df[df['Which layout? (write the layout name as it exactly appears on the repository)'].str.contains(layouts_prefix, na=False)]
    df = df[df['Reward'] != 'N/A']
    df = df[df['Trained on ... Layout(s)'].str.contains(trained_on_number_of_layouts, na=False)]
    df['Reward'] = pd.to_numeric(df['Reward'], errors='coerce')
    df = df.dropna(subset=['Reward'])

    unique_layouts = df['Which layout? (write the layout name as it exactly appears on the repository)'].unique()
    fig, axes = plt.subplots(nrows=len(unique_layouts), ncols=2, figsize=(27, 5 * len(unique_layouts)))

    if len(unique_layouts) == 1:
        axes = [axes]

    for i, layout in enumerate(unique_layouts):
        layout_df = df[df['Which layout? (write the layout name as it exactly appears on the repository)'] == layout]
        layout_df = layout_df.sort_values(by='Reward', ascending=False)
        layout_df['Label'] = layout_df.apply(
            lambda row: f"{row['Agent Type']}\n{row['LearnerType']}\n Trained on {row['Trained on ... Layout(s)']} layout(s)",
            axis=1
        )
        sns.barplot(x='Label', y='Reward', data=layout_df, ax=axes[i, 0], palette='viridis')
        axes[i, 0].set_title(f"Layout: {layout} - Sorted by Reward)")
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

    if layouts_prefix == '': layouts_prefix = 'all'
    if trained_on_number_of_layouts == '': trained_on_number_of_layouts = 'OneOrMultiple'

    plt.savefig(f'{QUESTIONNAIRES_DIR}/{questionnaire_file_name}/summary_in_{layouts_prefix}_layouts_trained_on_{trained_on_number_of_layouts}_layouts.png', dpi=100)


if __name__ == "__main__":
    questionnaire_file_name = '2'
    layouts_prefix = ''
    trained_on_number_of_layouts = '' # Multiple or One

    summarize_in_a_figure(questionnaire_file_name=questionnaire_file_name,
                          layouts_prefix=layouts_prefix,
                          trained_on_number_of_layouts=trained_on_number_of_layouts)