#----------------------------------------------------
# Imports
#----------------------------------------------------
import imports
import helper

#----------------------------------------------------
# Helper Functions for Evaluating Models
#----------------------------------------------------
def graph_feature_importance(model_name: str,
                             model, 
                             feature_names,
                             n: int=10) -> None:
    """
    Generates and saves a bar chart comparing the top feature importances

    model_name -> str
        The name of the model, used for naming the graphs

    model -> Sklearn.ensemble._forest.RandomForestClassifier
        The trained model that contains the feature weights

    feature_names -> Pandas.core.indexes.base.Index
        The names of the features

    n -> int
        Top n results shown in the feature comparison
        Default 10

    Returns -> None
        Graphs and shows the feature comparison
        Saves the feature comparison to Model folder

    Example
        graph_feature_importance(model, feature_names)
    """
    # Get feature importances
    feature_importances = model.feature_importances_

    # Get the top n feature importances and their corresponding names
    sorted_indices = imports.np.argsort(feature_importances)[::-1]
    top_indices = sorted_indices[:n]
    top_feature_names = feature_names[top_indices]
    top_feature_importances = feature_importances[top_indices]

    # Pre graph formatting
    imports.plt.figure(figsize=imports.DEFAULT_LONG_FIG_SIZE)
    imports.plt.title(f"Top {n} Most Relevant Features for {model_name}")
    imports.plt.xlabel("Feature Importance")
    imports.plt.yticks(range(n), top_feature_names)
    imports.plt.ylabel("Features")
    imports.plt.grid("True", alpha=imports.DEFAULT_GRID_ALPHA)

    # Graphing
    imports.plt.barh(range(n), top_feature_importances, align='center')
    imports.plt.gca().invert_yaxis()  # Invert y-axis to have the most important feature at the top

    # Post graph formatting
    imports.plt.tight_layout()
    filepath = f"{imports.DEV_PATH_TO_MODEL}/feature_importance_{model_name.replace(" ", "_").lower()}.png"
    imports.plt.savefig(filepath)
    imports.plt.show()

    return None

def graph_confusion_matrix(y_test: imports.pd.Series, 
                           y_pred: imports.pd.Series,
                           class_names: list,
                           model_name: str) -> imports.np.ndarray:
    """
    Generates and saves a confusion matrix from given y test,pred values

    y_test -> Pandas.Series
        The actual test values from the dependent variable

    y_pred -> Pandas.Series
        The predicted test values from the dependent variable

    class_names -> list
        List of all class names (typically positive and negative) for labeling confusion matrix

    model_name -> str
        The name of the model that produces the y test,pred values

    Returns -> Numpy.ndarray
        Graphs and shows the confusion matrix
        Saves the confusion matrix to Model folder
        Returns the confusion matrix as an np.ndarray

    Example
        graph_confusion_matrix(y_test, y_pred, ["class1", "class2", "class3"], model)
    """
    # Generating confusion matrix
    conf_matrix = imports.confusion_matrix(y_test, y_pred)

    # Pre graph formatting
    imports.plt.figure(figsize=imports.DEFAULT_LONG_FIG_SIZE)
    imports.plt.title(f"Confusion Matrix for {model_name}")

    # Graphing
    imports.sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                        xticklabels=class_names,
                        yticklabels=class_names)

    # Post graph formatting
    imports.plt.xlabel("Predicted Label")
    imports.plt.ylabel("True Label")
    imports.plt.tight_layout()
    filepath = f"{imports.DEV_PATH_TO_MODEL}/confusion_matrix_{model_name.replace(" ", "_").lower()}.png"
    imports.plt.savefig(filepath)
    imports.plt.show()

    return conf_matrix

def graph_classification_report(y_test: imports.pd.Series, 
                                y_pred: imports.pd.Series,
                                class_names: list,
                                model_name: str) -> imports.pd.DataFrame:
    """
    Generates and saves a classification report from given y test,pred values

    y_test -> Pandas.Series
        The actual test values from the dependent variable

    y_pred -> Pandas.Series
        The predicted test values from the dependent variable

    class_names -> list
        List of all class names (typically positive and negative) for labeling confusion matrix

    model_name -> str
        The name of the model that produces the y test,pred values

    Returns -> Pandas.DataFrame
        Graphs and shows the classification report
        Saves the classification report to Model folder
        Returns a pd.df of all the metrics

    Example
        graph_classification_report(y_test, y_pred)
    """
    # Generating classification report and converting
    report_dict = imports.classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    report_df = imports.pd.DataFrame(report_dict).transpose()

    # Heatmap visualization
    imports.plt.figure(figsize=imports.DEFAULT_LONG_FIG_SIZE)
    imports.sns.heatmap(report_df.iloc[:, :-1].T, annot=True, cmap='Blues')
    imports.plt.title(f"Classification Report for {model_name}")
    imports.plt.yticks(rotation=45)

    # Post Graph formating
    filepath = f"{imports.DEV_PATH_TO_MODEL}/class_report_{model_name.replace(" ", "_").lower()}.csv"
    report_df.to_csv(filepath)

    imports.plt.tight_layout()
    filepath = f"{imports.DEV_PATH_TO_MODEL}/class_report_{model_name.replace(" ", "_").lower()}.png"
    imports.plt.savefig(filepath)
    imports.plt.show()

    return report_df

def graph_binary_roc(y_test: imports.pd.Series,
                     y_pred_prob: imports.pd.Series,
                     model_name: str) -> imports.np.float64:
    """
    Generates and saves an ROC Curve from given y test,pred values

    y_test -> Pandas.Series
        The actual test values from the dependent variable

    y_pred_prob -> Pandas.Series
        The predicted test values as probabilities from the dependent variable

    model_name -> str
        The name of the model that produces the y test,pred values

    Returns -> Numpy.float64
        Graphs and shows the ROC Curve
        Saves the ROC Curve to Model folder
        Returns the AUC score

    Example
        graph_binary_roc(y_test, y_pred)
    """
    # Generating classification report and converting
    fpr, tpr, thresholds = imports.roc_curve(y_test, y_pred_prob)
    roc_auc = imports.auc(fpr, tpr)

    # Pre Graph formatting
    imports.plt.figure(figsize=imports.DEFAULT_SQUARE_FIG_SIZE)

    # Graphing
    roc_label = f"ROC curve (area = {round(roc_auc, 2)})"
    imports.plt.plot(fpr, tpr, color='blue', lw=2, label=roc_label)
    imports.plt.plot([0, 1], [0, 1], color='darkgray', lw=2, linestyle='--')

    # Post Graph formatting
    imports.plt.xlabel("False Positive Rate")
    imports.plt.ylabel("True Positive Rate")
    imports.plt.title(f"ROC Curve for {model_name}")
    imports.plt.legend(loc='lower right')
    imports.plt.xlim([0.0, 1.0])
    imports.plt.ylim([0.0, 1.05])
    imports.plt.tight_layout()
    filepath = f"{imports.DEV_PATH_TO_MODEL}/roc_{model_name.replace(" ", "_").lower()}.png"
    imports.plt.savefig(filepath)
    imports.plt.show()

    return roc_auc

def graph_multi_roc(class_names: list,
                    model,
                    y_test: imports.pd.Series,
                    y_pred_prob: imports.pd.Series,
                    model_name: str) -> imports.np.float64:
    """
    Generates and saves an multi-class ROC Curve using OvR strategy from given y test,pred values

    class_names -> list
        The list of class names
    
    model -> Sklearn.ensemble._forest.RandomForestClassifier
        Model to extract class values
        
    y_test -> Pandas.Series
        The actual test values from the dependent variable

    y_pred_prob -> Pandas.Series
        The predicted test values as probabilities from the dependent variable

    model_name -> str
        The name of the model that produces the y test,pred values

    Returns -> Numpy.float64
        Graphs and shows the ROC Curve
        Saves the ROC Curve to Model folder
        Returns the AUC score

    Example
        graph_multi_roc(y_test, y_pred)
    """
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    n_classes = len(class_names)
    for i in range(n_classes):
        fpr[i], tpr[i], _ = imports.roc_curve(y_test == model.classes_[i], y_pred_prob[:, i])
        roc_auc[i] = imports.auc(fpr[i], tpr[i])

    # Plot ROC curve for each class

    # Pre-Graph Formatting
    imports.plt.figure(figsize=imports.DEFAULT_SQUARE_FIG_SIZE)
    imports.plt.title(f"ROC Curve for {model_name}")
    imports.plt.xlabel("False Positive Rate")
    imports.plt.ylabel("True Positive Rate")
    imports.plt.grid("True", alpha=imports.DEFAULT_GRID_ALPHA)

    # Graphing
    imports.plt.plot([0, 1], [0, 1], "k--")
    imports.plt.xlim([0.0, 1.0])
    imports.plt.ylim([0.0, 1.05])

    for i in range(n_classes):
        imports.plt.plot(fpr[i], tpr[i], label=f"{class_names[i]} vs. Rest (AUC = {round(roc_auc[i], 5)})")

    # Post-Graphing Formatting
    imports.plt.legend(loc="lower right")
    imports.plt.tight_layout()
    filepath = f"{imports.DEV_PATH_TO_MODEL}/roc_curve_{model_name.replace(" ", "_").lower()}.png"
    imports.plt.savefig(filepath)
    imports.plt.show()

    return roc_auc

def graph_bar_comparison(values: imports.pd.DataFrame, 
                         title: str, 
                         xlabel: str, 
                         ylabel: str) -> None:
    """
    Generates and saves a multi-bar chart that compares all the values in the df.
    The columns in the df indicate the xticks while the rows indicate the legend labels. 

    values -> Pandas.DataFrame
        The given values to graph

    title -> str
        The given figure title

    xlabel -> str
        The given figure xlabel

    ylabel -> str
        The given figure ylabel

    Returns -> None
        Graphs and shows the bar chart

    Example
        graph_bar_comparison(df, "Title", "X", "Y")
    """
    # Pre Graph Formatting
    imports.plt.figure(figsize=imports.DEFAULT_LONG_FIG_SIZE)
    imports.plt.title(title)
    imports.plt.xlabel(xlabel)
    imports.plt.xticks(rotation=45)
    imports.plt.ylabel(ylabel)
    imports.plt.grid("True", alpha=imports.DEFAULT_GRID_ALPHA)

    all_values = list(values.values.flatten())
    imports.plt.ylim(helper.calc_limit_lwrbound(all_values))

    width = imports.DEFAULT_BAR_WIDTH
    ind = imports.np.arange(len(values.columns))
    imports.plt.xticks(ind + (width * 0.5 * (len(values) - 1)), labels=values.columns)

    # Graphing
    for i in range(len(values)):
        imports.plt.bar(ind + width * i, 
                        values.iloc[i].values,
                        width=width,
                        label=values.index[i],
                        alpha=imports.DEFAULT_GRAPH_ALPHA)

    # Post Graph Formatting & Saving
    imports.plt.tight_layout()
    imports.plt.legend(bbox_to_anchor=(1,1))
    filepath = f"{imports.DEV_PATH_TO_MODEL}/bar_comparisons_{title.replace(" ", "_").lower()}.png"
    imports.plt.savefig(filepath)
    imports.plt.show()

    return None

#----------------------------------------------------
# Main
#----------------------------------------------------
def main():
    return None

#----------------------------------------------------
# Entry
#----------------------------------------------------
if __name__ == "__main__":
    main()