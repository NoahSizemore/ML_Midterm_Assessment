"""Runs the whole project: predictions.csv first, then the feature-importance analysis."""
import feature_importance
import linear_gd

if __name__ == "__main__":
    linear_gd.main()
    feature_importance.main()
