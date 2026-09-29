# Model 2 - Stock-out Risk Classifier

## Why recall and PR-AUC matter most

A missed stock-out (false negative) means lost revenue: the shelf is empty and the sale never happens. A false alarm (false positive) triggers a phone call or a manual check. We optimise for recall and PR-AUC, not accuracy.

## Why accuracy misleads

With ~28% positives, always predicting no stock-out gives ~72% accuracy while missing every real event. PR-AUC has a floor equal to the base rate; it cannot be fooled by predicting the majority class.

## Imbalance handling

class_weight=balanced and scale_pos_weight up-weight the minority class. SMOTE was excluded: time-series structure means synthetic rows would mix features from different market conditions; at 28% positives the class ratio is manageable.

## Calibration

Class reweighting inflates raw probabilities above true event frequency. The business rules use fixed thresholds (>=0.70 HIGH, 0.40-0.70 MEDIUM). Calibration method chosen: isotonic. Brier score improved: 0.1657 -> 0.1588.

## Algorithm selection

XGBoost was selected because: (1) it handles NaN natively for products with sparse history; (2) it learns interactions (low stock AND high discount AND late supplier) that are the core drivers of stock-out risk; (3) the OOT coverage feature is leakage-free because the underlying forecast was made before those rows were seen. Logistic regression is not on the permitted algorithm list (DECISIONS D2.7).

## Algorithm comparison (Test set)

| Algorithm | PR-AUC | ROC-AUC | Recall@0.40 | F1@0.40 | Brier |
|---|---|---|---|---|---|
| LinearSVM(C=1.0) | 0.6489 | 0.8004 | 0.3738 | 0.4958 | 0.1400 |
| SELECTED+CAL: LinearSVM(C=1.0) | 0.6299 | 0.7988 | 0.4989 | 0.5561 | 0.1372 |
| LinearSVM(C=0.1) | 0.6293 | 0.7877 | 0.3949 | 0.5051 | 0.1413 |
| XGB(n=300,d=4,lr=0.05) | 0.6065 | 0.7800 | 0.5840 | 0.5435 | 0.1508 |
| XGB(n=300,d=4,lr=0.1) | 0.5921 | 0.7775 | 0.5750 | 0.5487 | 0.1514 |
| XGB(n=600,d=4,lr=0.05) | 0.5902 | 0.7669 | 0.5494 | 0.5335 | 0.1525 |
| XGB(n=300,d=6,lr=0.05) | 0.5813 | 0.7674 | 0.5283 | 0.5297 | 0.1524 |
| KNN(k=31) | 0.5802 | 0.7813 | 0.3941 | 0.4849 | 0.1461 |
| RF(n=200,d=10) | 0.5794 | 0.7705 | 0.5592 | 0.5367 | 0.1575 |
| RF(n=400,d=10) | 0.5790 | 0.7722 | 0.5629 | 0.5382 | 0.1571 |
| RF(n=400,d=20) | 0.5764 | 0.7706 | 0.3308 | 0.4414 | 0.1497 |
| RF(n=200,d=20) | 0.5700 | 0.7647 | 0.3338 | 0.4439 | 0.1505 |
| XGB(n=600,d=4,lr=0.1) | 0.5686 | 0.7609 | 0.5343 | 0.5221 | 0.1583 |
| KNN(k=15) | 0.5657 | 0.7725 | 0.5079 | 0.5431 | 0.1467 |
| XGB(n=300,d=6,lr=0.1) | 0.5623 | 0.7512 | 0.4921 | 0.5094 | 0.1583 |
| XGB(n=600,d=6,lr=0.05) | 0.5584 | 0.7471 | 0.4717 | 0.4964 | 0.1593 |
| XGB(n=600,d=6,lr=0.1) | 0.5526 | 0.7419 | 0.4461 | 0.4903 | 0.1650 |
| DecisionTree(d=5,ms=20) | 0.4980 | 0.7049 | 0.5087 | 0.4811 | 0.1701 |
| DecisionTree(d=5,ms=50) | 0.4980 | 0.7049 | 0.5087 | 0.4811 | 0.1701 |
| KNN(k=5) | 0.4788 | 0.7200 | 0.5494 | 0.5109 | 0.1703 |
| NaiveBayes(vs=1e-09) | 0.4258 | 0.7106 | 0.3052 | 0.3672 | 0.2422 |
| DecisionTree(d=10,ms=50) | 0.4163 | 0.6193 | 0.4514 | 0.4439 | 0.2105 |
| DecisionTree(d=10,ms=20) | 0.3902 | 0.6015 | 0.4461 | 0.4343 | 0.2224 |
| DecisionTree(d=None,ms=50) | 0.3337 | 0.6011 | 0.4295 | 0.3781 | 0.2610 |
| NaiveBayes(vs=0.001) | 0.3316 | 0.5991 | 0.2675 | 0.2927 | 0.2111 |
| DecisionTree(d=None,ms=20) | 0.2809 | 0.5564 | 0.3436 | 0.3325 | 0.2972 |
| Rule (stock<reorder) | 0.2697 | 0.5519 | 0.4393 | 0.3562 | 0.3909 |

## Segment analysis (Test set)

| Segment | Value | n | Pos% | PR-AUC | Recall@0.40 |
|---|---|---|---|---|---|
| category_name | Beverages | 756 | 54.6% | 0.839 | 0.816 |
| category_name | Dairy | 728 | 10.9% | 0.245 | 0.051 |
| category_name | Frozen | 364 | 21.7% | 0.524 | 0.418 |
| category_name | Groceries | 1680 | 20.2% | 0.482 | 0.250 |
| category_name | Household | 518 | 12.5% | 0.263 | 0.246 |
| category_name | Personal Care | 518 | 18.5% | 0.410 | 0.312 |
| category_name | Snacks | 826 | 30.9% | 0.714 | 0.616 |
| store_type_name | Express | 1344 | 21.4% | 0.548 | 0.436 |
| store_type_name | Hypermarket | 1106 | 32.1% | 0.729 | 0.625 |
| store_type_name | Supermarket | 2940 | 23.3% | 0.606 | 0.460 |
| promo_week | non-promo | 4274 | 20.1% | 0.515 | 0.367 |
| promo_week | promo | 1116 | 41.9% | 0.798 | 0.741 |

## Limitations

- Frozen and Snacks are the hardest segments (promotion-driven, volatile demand).
- The calibration set is only 15 days; isotonic regression can overfit small sets.
- XGBoost is not directly interpretable; permutation importance (see src/explain.py) gives the global picture; the what-if flip gives per-row explanations.