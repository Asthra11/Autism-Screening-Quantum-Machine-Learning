import plotly.graph_objects as go


def accuracy_bar_chart(results_dict):
    """
    Creates accuracy comparison bar chart
    results_dict = { model_name: accuracy }
    """

    fig = go.Figure()

    fig.add_bar(
        x=list(results_dict.keys()),
        y=[v * 100 for v in results_dict.values()],
        text=[f"{v * 100:.2f}%" for v in results_dict.values()],
        textposition="outside"
    )

    fig.update_layout(
        title="Model Accuracy Comparison",
        yaxis_title="Accuracy (%)",
        xaxis_title="Models",
        yaxis=dict(range=[0, 100]),
        template="plotly_white"
    )

    return fig


def confusion_matrix_heatmap(cm, model_name):
    """
    Creates confusion matrix heatmap
    """

    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=["Predicted No ASD", "Predicted ASD"],
            y=["Actual No ASD", "Actual ASD"],
            colorscale="Blues",
            showscale=True
        )
    )

    fig.update_layout(
        title=f"Confusion Matrix – {model_name}",
        xaxis_title="Prediction",
        yaxis_title="Actual"
    )

    return fig
