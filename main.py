# The necessary Python libraries for data manipulation and visualization are imported.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def main():
    data_raw = pd.read_csv('data-final.csv', sep='\t')
    data = data_raw.copy()
    pd.options.display.max_columns = 150  
    # Some columns of the dataset are dropped, and any rows containing NaN values are also dropped.
    data.drop(data.columns[50:107], axis=1, inplace=True)
    data.drop(data.columns[51:], axis=1, inplace=True)
    data.dropna(inplace=True)
    print('Number of participants: ', len(data))
    # Rename the columns: define new column names for out model for Ambitious and Methodical factors.
    new_column_names = [column for column in data]
    new_column_names[30] = "AMB1"
    new_column_names[31] = "MET1"
    new_column_names[32] = "MET2"
    new_column_names[33] = "MET3"
    new_column_names[34] = "AMB2"
    new_column_names[35] = "MET4"
    new_column_names[36] = "MET5"
    new_column_names[37] = "AMB3"
    new_column_names[38] = "MET6"
    new_column_names[39] = "AMB4"
    data.columns = new_column_names
    # Groups and Questions: The questions related to different traits like extroversion (EXT), neuroticism (EST),
    # agreeableness (AGR), conscientiousness (MET & AMB), and openness (OPN) are defined as dictionaries.
    ext_questions = {'EXT1': 'I am the life of the party',
                     'EXT2': 'I dont talk a lot',
                     'EXT3': 'I feel comfortable around people',
                     'EXT4': 'I keep in the background',
                     'EXT5': 'I start conversations',
                     'EXT6': 'I have little to say',
                     'EXT7': 'I talk to a lot of different people at parties',
                     'EXT8': 'I dont like to draw attention to myself',
                     'EXT9': 'I dont mind being the center of attention',
                     'EXT10': 'I am quiet around strangers'}
    est_questions = {'EST1': 'I get stressed out easily',
                     'EST2': 'I am relaxed most of the time',
                     'EST3': 'I worry about things',
                     'EST4': 'I seldom feel blue',
                     'EST5': 'I am easily disturbed',
                     'EST6': 'I get upset easily',
                     'EST7': 'I change my mood a lot',
                     'EST8': 'I have frequent mood swings',
                     'EST9': 'I get irritated easily',
                     'EST10': 'I often feel blue'}
    agr_questions = {'AGR1': 'I feel little concern for others',
                     'AGR2': 'I am interested in people',
                     'AGR3': 'I insult people',
                     'AGR4': 'I sympathize with others feelings',
                     'AGR5': 'I am not interested in other peoples problems',
                     'AGR6': 'I have a soft heart',
                     'AGR7': 'I am not really interested in others',
                     'AGR8': 'I take time out for others',
                     'AGR9': 'I feel others emotions',
                     'AGR10': 'I make people feel at ease'}
    met_questions = {
        'MET1': 'I leave my belongings around',
        'MET2': 'I pay attention to details',
        'MET3': 'I make a mess of things',
        'MET4': 'I often forget to put things back in their proper place',
        'MET5': 'I like order',
        'MET6': 'I follow a schedule'}
    amb_questions = {'AMB1': 'I am always prepared',
                     'AMB2': 'I get chores done right away',
                     'AMB3': 'I shirk my duties',
                     'AMB4': 'I am exacting in my work'}
    opn_questions = {'OPN1': 'I have a rich vocabulary',
                     'OPN2': 'I have difficulty understanding abstract ideas',
                     'OPN3': 'I have a vivid imagination',
                     'OPN4': 'I am not interested in abstract ideas',
                     'OPN5': 'I have excellent ideas',
                     'OPN6': 'I do not have a good imagination',
                     'OPN7': 'I am quick to understand things',
                     'OPN8': 'I use difficult words',
                     'OPN9': 'I spend time reflecting on things',
                     'OPN10': 'I am full of ideas'}
    # Group Names and Columns
    # EXT = [column for column in data if column.startswith('EXT')]
    # EST = [column for column in data if column.startswith('EST')]
    # AGR = [column for column in data if column.startswith('AGR')]
    MET = [column for column in data if column.startswith('MET')]
    AMB = [column for column in data if column.startswith('AMB')]

    # OPN = [column for column in data if column.startswith('OPN')]
    # Defining a function to visualize the questions and answers distribution: this function is defined to visually display
    # histograms of the distribution of answers for each question within a trait group.
    def vis_questions(groupname, questions, color):
        plt.figure(figsize=(40, 60))
        for i in range(1, len(groupname) + 1):
            plt.subplot(10, 5, i)
            plt.hist(data[groupname[i - 1]], bins=14, color=color, alpha=.5)
            plt.title(questions[groupname[i - 1]], fontsize=18)

    vis_questions(MET, met_questions, 'purple')
    vis_questions(AMB, amb_questions, 'green')
    # For ease of calculation lets scale all the values between 0-1 and take a sample of 5000
    df = data.drop('country', axis=1)
    columns = list(df.columns)
    scaler = MinMaxScaler(feature_range=(0, 1))  # The data is scaled using MinMaxScaler from scikit-learn to transform
    # features by scaling each feature to a given range (0 to 1 in this case). This is often done before applying machine
    # learning algorithms to ensure that all features have the same scale.
    df = scaler.fit_transform(df)
    df = pd.DataFrame(df, columns=columns)
    # I use the unscaled data but without the country column
    df_model = data.drop('country', axis=1)
    # define 6 clusters and fit the model
    kmeans = KMeans(n_clusters=6, n_init=10)  # Creating K-means Cluster Model
    k_fit = kmeans.fit(df_model)
    # Predicting the Clusters
    pd.options.display.max_columns = 10
    predictions = k_fit.labels_
    df_model['Clusters'] = predictions
    df_model.head()
    df_model.Clusters.value_counts()
    pd.options.display.max_columns = 150
    df_model.groupby('Clusters').mean()
    # Summing up the different questions groups
    col_list = list(df_model)
    ext = col_list[0:10]
    est = col_list[10:20]
    agr = col_list[20:30]
    met = col_list[31:34] + col_list[35:37] + col_list[38:39]
    amb = col_list[30:31] + col_list[34:35] + col_list[37:38] + col_list[39:40]
    opn = col_list[40:50]
    data_sums = pd.DataFrame()
    data_sums['extroversion'] = df_model[ext].sum(axis=1) / 10
    data_sums['neurotic'] = df_model[est].sum(axis=1) / 10
    data_sums['agreeable'] = df_model[agr].sum(axis=1) / 10
    data_sums['Methodicalness'] = df_model[met].sum(axis=1) / 6
    data_sums['ambition'] = df_model[amb].sum(axis=1) / 4
    data_sums['open'] = df_model[opn].sum(axis=1) / 10
    data_sums['clusters'] = predictions
    data_sums.groupby('clusters').mean()
    # Visualizing the means for each cluster
    dataclusters = data_sums.groupby('clusters').mean()
    plt.figure(figsize=(22, 5))  # Increase the height
    for i in range(0, 6):
        plt.subplot(1, 6, i + 1)
        plt.bar(dataclusters.columns, dataclusters.iloc[:, i], color='green', alpha=0.2)
        plt.plot(dataclusters.columns, dataclusters.iloc[:, i], color='red')
        plt.title('Cluster ' + str(i))
        plt.xticks(rotation=45, ha="right")  # Adjust rotation and alignment
        plt.ylim(0, 4)
    plt.tight_layout()  # Adjust the layout
    # In order to visualize in 2D graph I will use PCA
    pca = PCA(n_components=2)
    pca_fit = pca.fit_transform(df_model)
    df_pca = pd.DataFrame(data=pca_fit, columns=['PCA1', 'PCA2'])
    df_pca['Clusters'] = predictions
    df_pca.head()
    plt.figure(figsize=(10, 10))
    sns.scatterplot(data=df_pca, x='PCA1', y='PCA2', hue='Clusters', palette='Set2', alpha=0.8)
    plt.title('Personality Clusters after PCA')
    plt.show()

    # check correlation:

    # Calculate average scores for each factor for every individual
    data['EXT'] = data[list(ext_questions.keys())].mean(axis=1)
    data['EST'] = data[list(est_questions.keys())].mean(axis=1)
    data['AGR'] = data[list(agr_questions.keys())].mean(axis=1)
    data['MET'] = data[list(met_questions.keys())].mean(axis=1)
    data['AMB'] = data[list(amb_questions.keys())].mean(axis=1)
    data['OPN'] = data[list(opn_questions.keys())].mean(axis=1)

    # Now, compute the correlation matrix for these factors
    factors = ['EXT', 'EST', 'AGR', 'MET', 'AMB', 'OPN']
    correlation_matrix = data[factors].corr()

    # Visualize the correlation matrix using a heatmap
    plt.figure(figsize=(10, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Matrix of Personality Factors')
    plt.show()

    # correlation for the FFM

    # Compute conscientiousness as the average of MET and AMB
    data['CON'] = data[['MET', 'AMB']].mean(axis=1)

    # Compute the correlation matrix for these factors using original conscientiousness (CON)
    factors_with_con = ['EXT', 'EST', 'AGR', 'CON', 'OPN']
    correlation_matrix_with_con = data[factors_with_con].corr()

    # Visualize the correlation matrix using a heatmap for original conscientiousness (CON)
    plt.figure(figsize=(10, 10))
    sns.heatmap(correlation_matrix_with_con, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Matrix of Personality Factors with Original Conscientiousness')
    plt.show()


if __name__ == "__main__":
    main()