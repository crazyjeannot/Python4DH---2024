import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt


def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def vector_len(v):
    return np.sqrt(np.sum(v ** 2))

def cosine_distance(a, b):
    return 1 - np.dot(a, b) / (vector_len(a) * vector_len(b))

def get_distances(df, mots, distance):

    df_mots = df[mots]
    if distance == "euclidienne":
        distances = [[euclidean_distance(a, b) for b in df_mots.values] for a in df_mots.values]
    elif distance == "cosine":
        distances = [[cosine_distance(a, b) for b in df_mots.values] for a in df_mots.values]
    else:
        print("La distance choisie n'est pas la bonne")
        return None 

    df_dist = pd.DataFrame(distances, index=df_mots.index, columns=df_mots.index)

    distance_max = df_dist.max().max()
    distance_min = df_dist[df_dist > 0].min().min()  # Ignorer la diagonale principale
    
    index_max = df_dist.stack().idxmax()
    index_min = index_min = df_dist[df_dist > 0.000001].stack().idxmin()

    print(f"A/ Distance maximale entre deux romans : {distance_max}")
    print(f"   Couple de romans pour la distance maximale : {index_max}")

    print(f"B/ Distance minimale entre deux romans : {distance_min}")
    print(f"   Couple de romans pour la distance minimale : {index_min}")

    return df_dist 

def visualization(df_distances):

    plt.figure(figsize=(10, 8))
    plt.imshow(df_distances, cmap='Reds', interpolation='nearest')

    plt.colorbar(label='Similarité')
    plt.title("Distance Cosine entre nos textes d'après 100 mots")

    plt.xticks(range(len(df_distances.columns)), df_distances.columns, rotation=90)
    plt.yticks(range(len(df_distances.index)), df_distances.index)

    plt.show()



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--input', help='path to csv file', required=True
    )
    parser.add_argument(
        '-m', '--mot1', help='mot1', required=False
    )
    parser.add_argument(
        '-n', '--mot2', help='mot2', required=False
    )
    parser.add_argument(
        '-d', '--distance', help='distance cosine ou euclidienne', required=True
    )

    args = vars(parser.parse_args())
    inputcsv = args["input"]
    mot1 = args["mot1"]
    mot2 = args["mot2"]
    distance = args["distance"]

    df_main = pd.read_csv(inputcsv)

    
    df_main = df_main.fillna(0)
    df_main.set_index(['index'], inplace=True)


    mots = np.random.choice(df_main.columns, 100, replace=False)

    print(distance)
    if mot1 and mot2:
        df_distances = get_distances(df_main, [mot1, mot2], distance)

    else:
        print("random")
        df_distances = get_distances(df_main, mots, distance)


    visualization(df_distances)








