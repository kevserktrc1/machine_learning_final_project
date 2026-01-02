import matplotlib.pyplot as plt

def basic_info(df):
    print("\n--- Data Size ---")
    print(df.shape)

    print("\n--- Data Size, Data Types, and Fill/Empty Information---")
    print(df.info())

    print("\n--- Statistical Summary---")
    print(df.describe())

    print("\n---Missing Value Check ---")
    print(df.isnull().sum())


def plot_happiness_distribution(df, bins=25):
    df["Happiness_Index"].hist(bins=bins)
    plt.title("Distribution of Happiness Index")
    plt.xlabel("Happiness Index")
    plt.ylabel("n")
    plt.show()