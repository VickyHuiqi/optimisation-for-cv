import numpy as np

def iou(box, clusters):
    """
    Calculate the Intersection over Union (IoU) between a box and clusters.

    Args:
        box (np.ndarray): A 1D array containing the width and height of the box [width, height].
        clusters (np.ndarray): A 2D array of shape (n, 2) containing the width and height of n clusters.

    Returns:
        np.ndarray: A 1D array containing the IoU value between the box and each of the clusters.
    """
    # Calculate intersection
    intersection = np.minimum(clusters[:, 0], box[0]) * np.minimum(clusters[:, 1], box[1])
    # Calculate union
    box_area = box[0] * box[1]
    cluster_area = clusters[:, 0] * clusters[:, 1]
    union = box_area + cluster_area - intersection
    # Return IoU
    return intersection / union 

def kmeans(boxes, k):
    """
    Perform k-means clustering using the IoU distance metric.
    
    Args:
        boxes (np.ndarray): A 2D array of shape (n, 2), where n is the number of boxes, and each row contains the 
                             width and height of a box [width, height].
        k (int): The number of clusters (anchor boxes) to find.

    Returns:
        tuple: A tuple containing:
            - clusters (np.ndarray): A 2D array of shape (k, 2) with the final k cluster centers (anchor box dimensions).
            - nearest_clusters (np.ndarray): A 1D array of shape (n,) where each entry indicates the index of the nearest cluster for each box.
            - distances (np.ndarray): A 2D array of shape (n, k) containing the distance (1 - IoU) between each box and each cluster.
    """
    np.random.seed(seed=42)
    rows = boxes.shape[0]

    # Initialize distances and clusters
    distances = np.empty((rows, k))
    last_clusters = np.zeros((rows,))

    # Randomly initialize cluster centers
    clusters = boxes[np.random.choice(rows, k, replace=False)]

    while True:
        # Assign each box to the nearest cluster
        for i in range(k):
            distances[:, i] = 1 - iou(clusters[i], boxes)
        nearest_clusters = np.argmin(distances, axis=1)

        # Stop if clusters do not change
        if (last_clusters == nearest_clusters).all():
            break

        # Update cluster centers
        for i in range(k):
            clusters[i] = np.mean(boxes[nearest_clusters == i], axis=0)

        # Update cluster centers to medoids (box with max average IoU in the cluster)
        #for i in range(k):
        #    cluster_boxes = boxes[nearest_clusters == i]
        #    if len(cluster_boxes) == 0:
        #        continue
        #    ious = np.array([iou(box, cluster_boxes).mean() for box in cluster_boxes])
        #    clusters[i] = cluster_boxes[np.argmax(ious)]

        last_clusters = nearest_clusters

    return clusters, nearest_clusters, distances

