# Introduction

The feature engineering strategy focuses on creating two enriched views that capture the multi-dimensional aspects of user-item interactions in the Sheetz convenience store context. The design supports the NCF model's dual learning paths (Matrix Factorization and Multi-Layer Perceptron) while incorporating temporal and categorical hierarchies.

# User Features

For the user features view, we engineered three key feature categories:

### Sequential Behavior Patterns:
The user_sequences CTE captures the last 50 interactions for each user, ordered by time. This is necessary for the model's attention mechanism, allowing it to learn both short-term and long-term preference patterns. Rather than treating each purchase as independent, this sequential representation enables the model to understand purchase dependencies and temporal relationships.

### Category Preferences: 
Through the category_preferences CTE, we aggregate users' interactions with different product categories using a two-step aggregation process. This addresses the hierarchical nature of product relationships - from department level down to specific categories. This hierarchical understanding is vital for the model's ability to make cross-category recommendations and understand category substitutions.

### Temporal Patterns: 
The temporal feature engineering (through hourly_counts and daily_counts) captures cyclic patterns in user behavior. By separating hourly and daily patterns, we enable the model to learn both micro-patterns (time-of-day preferences) and macro-patterns (day-of-week routines). This is particularly important for a convenience store context where timing significantly influences purchase decisions.

# Product Features

For the product features view, our engineering focuses on:

### Co-occurrence Patterns: 
The pair_counts and product_pairs CTEs capture product relationships through actual purchase co-occurrences. This information is needed for the NCF model's ability to learn complementary product relationships and make contextually relevant recommendations. The two-step aggregation process allows us to maintain the top 10 strongest relationships for each product while calculating meaningful affinity scores.

### Purchase Dynamics: 
Through product_stats, we capture both popularity metrics (total_purchases) and engagement metrics (unique_customers). The ratio between these (purchase_loyalty_score) helps the model understand product stickiness and repeat purchase patterns. This is particularly valuable for the model's ability to balance between popular items and personalized recommendations.

### Temporal Distributions: 
Similar to user patterns, we capture product purchase timing through hourly_sales_counts and daily_sales_counts. This is needed for understanding product-specific temporal patterns (e.g., coffee in the morning, snacks in the afternoon) and enables the model to make time-aware recommendations.

# Architectural Considerations

The architectural decisions in both views were influenced by several key considerations:

### BigQuery Optimization: 
The use of separate CTEs for counting and aggregation avoids nested aggregations while maintaining query efficiency. This is particularly important for real-time inference where feature computation needs to be fast.

### Feature Granularity: 
We maintain both granular features (individual purchase records) and aggregated features (preference summaries). This allows the NCF model's different components to operate at appropriate levels of abstraction - the MLP can learn from aggregated patterns while the attention mechanism can focus on specific sequence details.

### Scalability: 
The views are designed to handle growing data volumes efficiently through appropriate partitioning and pre-aggregation. This is crucial for maintaining model performance as the dataset grows.

# Conclusion

These enriched views serve multiple purposes in the NCF architecture:

- The user embeddings can learn from both static features (demographics) and dynamic features (temporal patterns)
- The product embeddings can capture both intrinsic properties and relational properties (co-purchase patterns)
- The attention mechanism can leverage sequential purchase history to capture evolving preferences
- The temporal encodings can learn from both user-specific and product-specific timing patterns

This comprehensive feature engineering approach enables the NCF model to capture the complex, multi-dimensional nature of convenience store shopping behavior while maintaining computational efficiency and scalability.