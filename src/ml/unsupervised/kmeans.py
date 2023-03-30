from src.core.algebra import vector as v
from typing import TypeVar, Generic, List, Callable, Tuple, Dict
import numpy as np
from collections import defaultdict
T = TypeVar('T')

VectorSet = List[v.Vector[T]]
Cluster = List[v.Vector[T]]


class KMeans(Generic[T]):
    def __init__(
        self,
        k: int, tol: float,
        initialize_centroids: Callable[[VectorSet], VectorSet],
        distance: Callable[[v.Vector[T], v.Vector[T]], float],
        select_centroid: Callable[[Cluster], v.Vector[T]]
    ) -> None:
        self.k = k
        self.tol = tol
        self.fi_centroids = initialize_centroids
        self.f_distance = distance
        self.select_centroid = select_centroid

    def fit(self, x: v.VectorialCollection[T]):

        centroids = self.fi_centroids(x)

        count, inertia, distance = 0, 0, 0
        clusters: Dict[int, Cluster] = defaultdict(list)
        while count == 0 or distance <= self.tol:

            for vector in x:
                _, index_centroid = min([
                    (self.f_distance(centroid, vector), c)
                    for c, centroid in enumerate(centroids)
                ])

                clusters[index_centroid].append(vector)

            updated_centroids = [
                self.select_centroid(clusters[key_cluster])
                for key_cluster in range(self.k)
            ]

            changes, distance = 0, 0
            for i in range(self.k):
                if not updated_centroids[i] == centroids[i]:
                    changes += 1
                    distance += self.f_distance(
                        updated_centroids[i], centroids[i]
                    )

            count += changes
            inertia += distance
            centroids = updated_centroids

        self.centroids = centroids
        self.clusters = clusters

        result = []
        for vector in x:
            for cluster_tag, cluster in enumerate(clusters.values()):
                if vector in cluster:
                    result.append(cluster_tag)
                    break

        return result
