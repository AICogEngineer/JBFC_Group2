# Planning: Using ChromaDB & Embeddings for Image Classification Support

## Chromadb goals

## Things to Explore
- How can embedding similarity help classification?

---ANS: compare images based on visual features rather than fixed class outputs
---It improves robustness when classes have visual overlap
---Interpretable results (near neighbors)
--- works even when CNN is uncertain or undertraind

- Should embeddings be used:
  - After CNN? <-- uses CNN as feature extractor, store feature vectors in chromadb. classify images via nearestneighbor similarity isntead of softmax. Enables flexible class additions and avoids retraining on small changes

  - Alongside CNN? <-- CNN makes a class prediction. embeddings verify or refine that predicition with similiarity. Useful for confidence estimation and error detection

- What problems can embeddings solve that softmax alone cannot?
---ANS: class imbalance (less bias towards dominant classes), new classes without retraining, interpretability, outliers, few-shot examples (small amount per certain classes)

## Possible Approaches to Research

### Similarity Search (Baseline Use Case)
**Idea**
- For a new image:
  - Embed it
  - Query ChromaDB for top-k most similar embeddings
  - Use majority vote or weighted similarity for class prediction

**Things to reference in notes**
- Cosine vs Euclidean similarity
- Choosing `k`
- Confidence scoring from distances

**Why Useful**
- Simple, interpretable
- Works even without a trained CNN
- Good fallback if CNN is uncertain

### CNN + Embedding Hybrid Classification
**Idea**
- CNN extracts feature vectors (before final dense layer)
- Store those vectors in ChromaDB
- Classification = similarity search instead of softmax

**Things to Learn**
- Removing / bypassing final softmax layer
- Using CNN as a feature extractor
- Nearest-neighbor classification

**Why Useful**
- Handles class imbalance better
- Easier to add new classes without retraining
- More flexible than fixed-output CNN

### Post-Prediction Verification
**Idea**
- CNN predicts a class
- Embedding similarity checks if it actually looks like other images in class
- Flag low-similarity predictions as uncertain

**Things to Learn**
- Distance thresholds
- Confidence heuristics
- Outlier detection

**Why Useful**
- Reduces silent misclassifications
- Helps with noisy or ambiguous images

### Dataset Exploration & Debugging
**Idea**
- Use ChromaDB to:
  - Find mislabeled images
  - Find duplicates
  - Find images that don’t match their class cluster

**Things to Learn**
- Visualizing nearest neighbors
- Cluster consistency per class
- Average intra-class distances

**Why Useful**
- Improves dataset quality
- Helps explain model failures
- Good for project reports

## Metrics & Evaluation Ideas
- Accuracy
- Confusion matrices
- Top-k accuracy

## Next steps (december 30?)
- Research:
  - “Embedding-based image classification”
  - “Nearest neighbor classification with CNN embeddings”
- Prototype:
  - Query top-5 similar images from ChromaDB
  - Print their classes and distances
- Document:
  - Pros/cons of each approach
  - When to prefer CNN vs embeddings
