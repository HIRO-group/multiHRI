import numpy as np

import numpy as np

def create_latex_table(methods, chart_data):
    # Start of the LaTeX table
    latex_table = []
    latex_table.append("\\begin{table}[htbp]")
    latex_table.append("\\centering")
    latex_table.append("\\small")
    
    # Calculate the tabular format based on number of methods
    tabular_format = "l|" + "c" * len(methods)
    latex_table.append("\\begin{tabular}{" + tabular_format + "}")
    latex_table.append("\\hline")
    
    # Add headers (methods)
    latex_table.append("Layout & " + " & ".join(methods) + " \\\\")
    latex_table.append("\\hline")
    
    # Get layouts (using first method since they're the same for all)
    layouts = chart_data[methods[0]]['layout']
    
    # Store means for average calculation
    all_means = {method: [] for method in methods}
    
    # Add data for each layout
    for layout_idx, layout in enumerate(layouts):
        row_values = []
        for method in methods:
            mean = chart_data[method]['mean'][layout_idx]
            std = chart_data[method]['std'][layout_idx]
            all_means[method].append(mean)
            row_values.append(f"{mean:.2f} $\\pm$ {std:.2f}")
        
        # Add the row
        latex_table.append(f"{layout} & " + " & ".join(row_values) + " \\\\")
    
    # Add horizontal line before average row
    latex_table.append("\\hline")
    
    # Calculate and add average row
    avg_values = []
    for method in methods:
        method_mean = np.mean(all_means[method])
        # Calculate standard error of the mean
        method_std = np.std(all_means[method]) / np.sqrt(len(all_means[method]))
        avg_values.append(f"{method_mean:.2f} $\\pm$ {method_std:.2f}")
    
    latex_table.append("\\textbf{Average} & " + " & ".join(avg_values) + " \\\\")
    
    # Close the table
    latex_table.append("\\hline")
    latex_table.append("\\end{tabular}")
    latex_table.append("\\caption{Performance comparison across different methods}")
    latex_table.append("\\label{tab:performance}")
    latex_table.append("\\end{table}")
    
    return "\n".join(latex_table)


# # Your data
# methods = ['sALMH 6s', 'dsALMH 6d[2t] 6s', 'dsALMH 2d[2t] 2s', 'FCP_s1010_h256', 'SP_s1010_h256']
# chart_data = {'SP_s1010_h256': {'mean': [np.float64(33.53333333333333), np.float64(114.93333333333334), np.float64(412.93333333333334), np.float64(553.9333333333333)], 'std': [np.float64(12.977012729610303), np.float64(30.116189595943194), np.float64(84.50589050738826), np.float64(35.2880751810144)], 'layout': ['Counter Circuit', 'Resource Corridor', 'Secret Resources', 'No Counter Space']}, 'FCP_s1010_h256': {'mean': [np.float64(36.4), np.float64(110.53333333333335), np.float64(376.5333333333333), np.float64(398.20000000000005)], 'std': [np.float64(20.249596340663633), np.float64(18.456419167481954), np.float64(60.4194410036713), np.float64(39.423900786439454)], 'layout': ['Counter Circuit', 'Resource Corridor', 'Secret Resources', 'No Counter Space']}, 'dsALMH 2d[2t] 2s': {'mean': [np.float64(57.46666666666667), np.float64(112.93333333333334), np.float64(481.59999999999997), np.float64(560.0666666666667)], 'std': [np.float64(15.034492073317457), np.float64(20.209551858693597), np.float64(76.26698808044812), np.float64(36.215015990227464)], 'layout': ['Counter Circuit', 'Resource Corridor', 'Secret Resources', 'No Counter Space']}, 'dsALMH 6d[2t] 6s': {'mean': [np.float64(73.73333333333333), np.float64(117.93333333333332), np.float64(426.5333333333333), np.float64(575.1333333333333)], 'std': [np.float64(17.17847503902536), np.float64(24.15190145101776), np.float64(64.31508138084858), np.float64(36.244594800716385)], 'layout': ['Counter Circuit', 'Resource Corridor', 'Secret Resources', 'No Counter Space']}, 'sALMH 6s': {'mean': [np.float64(119.93333333333334), np.float64(133.79999999999998), np.float64(392.59999999999997), np.float64(580.8666666666667)], 'std': [np.float64(26.73801431648381), np.float64(19.441073205464516), np.float64(61.591039346144385), np.float64(34.92874345312604)], 'layout': ['Counter Circuit', 'Resource Corridor', 'Secret Resources', 'No Counter Space']}}

# latex_output = create_latex_table(methods, chart_data)
# print(latex_output)


import numpy as np

# Define the data
# data = {
#     'CAP_best': [119.93, 133.80, 481.60, 580.87],
#     'CAP_6d_6s': [73.73, 117.93, 426.53, 575.13],
#     'FCP': [36.40, 110.53, 376.53, 398.20],
#     'SP': [33.53, 114.93, 412.93, 553.93]
# }

# # Calculate mean and std for each column
# for column, values in data.items():
#     values_array = np.array(values)
#     mean = np.mean(values_array)
#     std = np.std(values_array, ddof=1)  # ddof=1 for sample standard deviation
#     print(f"{column}:")
#     print(f"Mean: {mean:.3f}")
#     print(f"Std: {std:.3f}")
#     print(f"Mean ± Std: {mean:.3f} ± {std:.3f}")
#     print()