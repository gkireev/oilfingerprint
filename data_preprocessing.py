import pandas as pd

def preprocessing (data):

    # removing insignificant data
    percent = 0.05
    freq = data["Oil group"].value_counts()
    signif = freq[freq / len(data) >= percent]
    mask = data["Oil group"].isin(list(signif.index))
    data = data[mask]

    data = data.drop("Wells", axis = 1)

    # scaling the Depth into range 0 to 1
    data = data.rename(columns={"Depth (m)": "Depth"})
    data["Depth"] = (data["Depth"] - data["Depth"].min()) / (
        data["Depth"].max() - data["Depth"].min()
    )

    # variables separation
    data_pred = data["Oil group"]
    data = data.drop("Oil group", axis=1)

    # feature engineering
    data = pd.get_dummies(data)
    X, y = data.values, data_pred.values

    return X, y