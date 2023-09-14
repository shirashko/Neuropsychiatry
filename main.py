# The necessary Python libraries for data manipulation and visualization are imported.
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def main():
    # loading data and preprocessing
    data_raw = pd.read_csv('data-final.csv', sep='\t')
    data = data_raw.copy()
    pd.options.display.max_columns = 150
    # Some columns of the dataset are dropped, and any rows containing NaN values are also dropped.
    data.drop(data.columns[50:107], axis=1, inplace=True)
    data.drop(data.columns[51:], axis=1, inplace=True)
    data.dropna(inplace=True)
    data = data.drop('country', axis=1)
    data = data[(data > 1).any(axis=1)]
    data = data[(data < 5).any(axis=1)]

    CON = data.iloc[:, 30:40].mean(axis=1)  # compute for showing correlation in big 5 later

    # process data to fit the 6 factor model we suggest
    # Rename the columns: define new column names for our model for Ambitious and Methodical factors.
    new_column_names = [column for column in data]
    new_column_names[30] = "AMB1"
    new_column_names[31] = "MET1"
    new_column_names[32] = "MET2"
    new_column_names[33] = "MET3"
    new_column_names[34] = "AMB2"
    new_column_names[35] = "MET4"
    new_column_names[36] = "MET5"
    new_column_names[37] = "MET6"
    new_column_names[38] = "AMB3"
    new_column_names[39] = "MET7"
    data.columns = new_column_names

    # Define trait groups and their corresponding questions
    trait_groups = {
        'EXT': ['EXT1', 'EXT2', 'EXT3', 'EXT4', 'EXT5', 'EXT6', 'EXT7', 'EXT8', 'EXT9', 'EXT10'],
        'EST': ['EST1', 'EST2', 'EST3', 'EST4', 'EST5', 'EST6', 'EST7', 'EST8', 'EST9', 'EST10'],
        'AGR': ['AGR1', 'AGR2', 'AGR3', 'AGR4', 'AGR5', 'AGR6', 'AGR7', 'AGR8', 'AGR9', 'AGR10'],
        'MET': ['MET1', 'MET2', 'MET3', 'MET4', 'MET5', 'MET6', 'MET7'],
        'AMB': ['AMB1', 'AMB2', 'AMB3'],
        'OPN': ['OPN1', 'OPN2', 'OPN3', 'OPN4', 'OPN5', 'OPN6', 'OPN7', 'OPN8', 'OPN9', 'OPN10']
    }

    # Create a data table with 6 trait columns, each calculated as the average answer for questions related to this trait
    trait_data = pd.DataFrame()
    for trait, questions in trait_groups.items():
        trait_data[trait] = data[questions].mean(axis=1)


    # Apply MinMax scaling to the trait data features by scaling each feature to a given range (0 to 1 in this case).
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_trait_data = scaler.fit_transform(trait_data)
    scaled_trait_data = pd.DataFrame(scaled_trait_data, columns=trait_data.columns)

    # Perform k-means clustering - define 6 clusters and fit the model
    kmeans = KMeans(n_clusters=6, random_state=0, n_init=10)
    cluster_labels = kmeans.fit_predict(scaled_trait_data) + 1  # Adjust cluster labels

    # Add cluster labels to scaled trait data
    scaled_trait_data['Clusters'] = cluster_labels

    # Calculate the mean trait values for each cluster
    mean_trait_clusters = scaled_trait_data.groupby('Clusters').mean()

    # Create subplots
    fig, axs = plt.subplots(1, 6, figsize=(22, 5), sharey=True)
    for cluster_id in range(1, 7):  # Adjust cluster range
        ax = axs[cluster_id - 1]
        cluster_data = mean_trait_clusters.loc[cluster_id]
        cluster_data.plot(kind='bar', ax=ax, color='green', alpha=0.2)
        cluster_data.plot(kind='line', ax=ax, color='red')
        ax.set_title('Cluster ' + str(cluster_id))
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        ax.set_ylim(0, 1)  # Adjust y-axis limits
        ax.set_ylabel('Mean Trait Value')

    plt.tight_layout()  # Adjust the layout
    plt.show()

    # Apply PCA on the clusters
    pca = PCA(n_components=2)
    pca_fit = pca.fit_transform(trait_data)
    pca_data = pd.DataFrame(data=pca_fit, columns=['PCA1', 'PCA2'])

    # Add adjusted cluster labels to PCA data
    pca_data['Clusters'] = cluster_labels

    # Visualization
    plt.figure(figsize=(10, 10))
    sns.scatterplot(data=pca_data, x='PCA1', y='PCA2', hue='Clusters', palette='Set2', alpha=0.8)
    plt.title('Personality Clusters after PCA')
    plt.show()
    
    # check correlation

    # Calculate the correlation matrix for the factors
    factors = ['EXT', 'EST', 'AGR', 'MET', 'AMB', 'OPN']
    correlation_matrix = trait_data[factors].corr()

    # Visualize the correlation matrix using a heatmap
    plt.figure(figsize=(10, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Matrix of Personality Factors')
    plt.show()

    # Compute the correlation matrix for factors with original conscientiousness (CON)
    trait_data_big_five = trait_data.drop(['MET', 'AMB'], axis=1)
    trait_data_big_five['CON'] = CON
    factors_with_con = ['EXT', 'EST', 'AGR', 'CON', 'OPN']
    correlation_matrix_with_con = trait_data_big_five[factors_with_con].corr()

    # Visualize the correlation matrix using a heatmap for original conscientiousness (CON)
    plt.figure(figsize=(10, 10))
    sns.heatmap(correlation_matrix_with_con, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Matrix of Personality Factors with Original Conscientiousness')
    plt.show()


if __name__ == "__main__":
    main()
