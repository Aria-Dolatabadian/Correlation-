import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Step 1: Reading the CSV file
data = pd.read_csv("TEST.csv")

# Step 2: Ensure the dataset contains only numeric columns
data = data.select_dtypes(include=[np.number])

# Step 3: Pearson correlation and p-values calculation
cor_coeff = data.corr(method='pearson')
p_values = pd.DataFrame(np.zeros(cor_coeff.shape), columns=cor_coeff.columns, index=cor_coeff.index)

for i in range(len(cor_coeff.columns)):
    for j in range(len(cor_coeff.columns)):
        if i != j:
            _, p = pearsonr(data.iloc[:, i], data.iloc[:, j])
            p_values.iloc[i, j] = p

# Handle missing values in the correlation matrix (replace NA with 0 for visualization purposes)
cor_coeff.fillna(0, inplace=True)
p_values.fillna(1, inplace=True)  # Set NA p-values to 1 (non-significant)

# Export the correlation coefficients and p-values to CSV
cor_coeff.to_csv("correlation_coefficients.csv")
p_values.to_csv("p_values.csv")

# Step 4: Prepare the correlation matrix for visualization
cor_coeff.values[np.diag_indices_from(cor_coeff)] = np.nan  # Remove self-correlations
cor_melt = cor_coeff.stack().reset_index()
cor_melt.columns = ["Trait1", "Trait2", "Correlation"]
cor_melt = cor_melt[cor_melt["Trait1"] != cor_melt["Trait2"]]  # Remove self-correlations
cor_melt = cor_melt[~cor_melt.set_index(['Trait1', 'Trait2']).index.duplicated(keep='first')]  # Remove duplicate correlations
cor_melt = cor_melt[abs(cor_melt["Correlation"]) > 0.1]  # Filter out near-zero correlations

# Define correlation limits for color transparency
cmap = sns.diverging_palette(240, 10, as_cmap=True)

# Set transparency based on correlation magnitude
cor_melt['Transparency'] = 1 - abs(cor_melt['Correlation'])

# Step 5: Define custom sector colors for each trait
sector_colors = {
    'Trait1': "#FF6347", 'Trait2': "#4682B4", 'Trait3': "#3CB371", 'Trait4': "#FFD700",
    'Trait5': "#FF69B4", 'Trait6': "#87CEEB", 'Trait7': "#9370DB"
}
sector_colors = {trait: sector_colors.get(trait, "#000000") for trait in pd.concat([cor_melt["Trait1"], cor_melt["Trait2"]]).unique()}

# Step 6: Create the heatmap
plt.figure(figsize=(10, 8))
heatmap = sns.heatmap(cor_coeff, annot=True, fmt=".2f", cmap=cmap, mask=cor_coeff.isnull(), square=True,
                       cbar_kws={"shrink": .8}, linewidths=0.5)

# Set plot labels and title
plt.title('Correlation Heatmap', fontsize=16)
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()

# The colorbar is already part of the heatmap object; you can customize it here if needed
cbar = heatmap.collections[0].colorbar
cbar.set_label('Correlation', rotation=270, labelpad=20)

# Step 8: Export the plot as a high-resolution image
plt.savefig("heatmap_with_legend_high_res.png", dpi=300)
plt.show()
