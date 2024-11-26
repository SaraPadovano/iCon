from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from kneed import KneeLocator
import seaborn as sns

# Funzione ch visualizza il grafico per il miglior k
def plot_k(maxK, inertia, file, kl):
    plt.figure(figsize=(8, 6))
    plt.plot(maxK, inertia, marker='o', linestyle='-', color='b', label='Inertia')
    plt.scatter(kl.elbow, inertia[kl.elbow - 1], c='red', s=100, label=f'Miglior k: {kl.elbow}')
    plt.xlabel('Numero di clusters')
    plt.ylabel('Inertia')
    plt.title('Metodo del gomito per il k ottimale')
    plt.legend()
    plt.grid(False)
    plt.savefig(file)
    plt.close()

# Funzione che calcola il numero di cluster ottimale per il dataset mediante il metodo del gomito
def regola_gomito(dataset, file):
    inertia = []
    # Fisso un range di k da 1 a 10
    k_range = range(1,11)
    for k in k_range:
        #random restart
        kmeans = KMeans(n_clusters=k, n_init=5, init='random')
        kmeans.fit(dataset)
        inertia.append(kmeans.inertia_)
    kl = KneeLocator(k_range, inertia, curve="convex", direction="decreasing")
    plot_k(k_range, inertia, file, kl)
    return kl.elbow

# Visualizza un grafico a torta con la distribuzione delle macchine nei clusters
def visualize_chart_cars(dataFrame, filec, titlePrefix=''):
    # Conteggio dei cluster
    cluster_counts = dataFrame['cluster'].value_counts()

    # Colori dinamici
    n_clusters = len(cluster_counts)
    colors = sns.color_palette('tab10', n_clusters)

    # Creazione del grafico
    plt.figure(figsize=(8, 8))
    wedges, texts, autotexts = plt.pie(
        cluster_counts,
        labels=cluster_counts.index,
        autopct=lambda p: f'{p:.1f}%\n({int(p / 100 * sum(cluster_counts))})',  # Percentuali e valori assoluti
        startangle=140,
        colors=colors
    )

    # Titolo dinamico
    plt.title(f'{titlePrefix} Distribuzione delle macchine nei clusters', fontsize=14, fontweight='bold')

    # Legenda
    sorted_counts = cluster_counts.sort_index()  # Ordina i cluster nella legenda
    plt.legend(
        wedges,
        [f'Cluster {i}: {count} items' for i, count in sorted_counts.items()],
        title="Clusters",
        loc="lower left",
        bbox_to_anchor=(1, 0, 0.5, 1)  # Posiziona la legenda fuori dal grafico
    )
    plt.tight_layout()  # Ottimizza il layout
    plt.savefig(filec, dpi=300)
    plt.close()

# Funzione che esegue il kmeans e restituisce le etichette e i centroidi
def cluster(df, features, filek, filec, fileName_clusters, titlePrefix=''):
    y = df[features]
    clusters = regola_gomito(y, filek)

    # Crea un modello con il numero ottimale di clusters
    model = KMeans(n_clusters=clusters, n_init=10, init='random')
    model.fit(y)

    # Aggiunge la label clusters e crea un nuovo file csv
    clusters = model.labels_
    centroids = model.cluster_centers_

    df['cluster'] = model.labels_
    df.to_csv(fileName_clusters, index=False)

    visualize_chart_cars(df, filec, titlePrefix)

    return clusters, centroids
