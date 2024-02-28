# Cf. p. 432
import logging
import pandas as pd
import sklearn
import sklearn.model_selection
import torch
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader, TensorDataset

logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s %(message)s')

def main():
    logging.info("prediction_fuel_efficiency.main()")

    url = "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
    column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration',
                    'Model Year', 'Origin']
    df = pd.read_csv(url, names=column_names, na_values="?", comment='\t', sep=" ", skipinitialspace=True)

    # Drop the NA rows
    df = df.dropna()
    df = df.reset_index(drop=True)

    # Train/test split
    df_train, df_test = sklearn.model_selection.train_test_split(
        df, train_size=0.8, random_state=1
    )
    train_stats = df_train.describe().transpose()
    logging.info(f"train_stats:\n{train_stats}")

    numeric_column_names = ['Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration']
    df_train_norm, df_test_norm = df_train.copy(), df_test.copy()
    for col_name in numeric_column_names:
        mean = train_stats.loc[col_name, 'mean']
        std = train_stats.loc[col_name, 'std']
        df_train_norm.loc[:, col_name] = (df_train_norm.loc[:, col_name] - mean)/std
        df_test_norm.loc[:, col_name] = (df_test_norm.loc[:, col_name] - mean) / std
    logging.info(f"df_train_norm.tail():\n{df_train_norm.tail()}")

    # Bucketize the model year
    boundaries = torch.tensor([73, 76, 79])
    v = torch.tensor(df_train_norm['Model Year'].values)
    df_train_norm['Model Year Bucketed'] = torch.bucketize(
        v, boundaries, right=True
    )
    v = torch.tensor(df_test_norm['Model Year'].values)
    df_test_norm['Model Year Bucketed'] = torch.bucketize(
        v, boundaries, right=True
    )
    numeric_column_names.append('Model Year Bucketed')

    # One-hot encoding of the origin country
    total_origin = len(set(df_train_norm['Origin']))  # 3
    origin_encoded = one_hot(torch.from_numpy(df_train_norm['Origin'].values) % total_origin)
    x_train_numeric = torch.tensor(df_train_norm[numeric_column_names].values)
    x_train = torch.cat([x_train_numeric, origin_encoded], 1).float()

    origin_encoded = one_hot(torch.from_numpy(df_test_norm['Origin'].values) % total_origin)
    x_test_numeric = torch.tensor(df_test_norm[numeric_column_names].values)
    x_test = torch.cat([x_test_numeric, origin_encoded], 1).float()

    y_train = torch.tensor(df_train_norm['MPG'].values).float()
    y_test = torch.tensor(df_test_norm['MPG'].values).float()

    # Create a dataset and a dataloader
    train_ds = TensorDataset(x_train, y_train)
    batch_size = 8
    torch.manual_seed(1)
    train_dl = DataLoader(train_ds, batch_size, shuffle=True)

    # Create the model
    hidden_units = [8, 4]
    input_size = x_train.shape[1]
    all_layers = []
    for hidden_unit in hidden_units:
        layer = torch.nn.Linear(input_size, hidden_unit)
        all_layers.append(layer)
        all_layers.append(torch.nn.ReLU())
        input_size = hidden_unit
    all_layers.append(torch.nn.Linear(hidden_units[-1], 1))
    model = torch.nn.Sequential(*all_layers)

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    torch.manual_seed(1)
    num_epochs = 200
    log_epochs = 20

    for epoch in range(num_epochs):
        loss_hist_train = 0
        for x_batch, y_batch in train_dl:
            pred = model(x_batch)[:, 0]
            loss = loss_fn(pred, y_batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_hist_train += loss.item()
        if epoch % log_epochs == 0:
            logging.info(f"Epoch {epoch}  Loss {loss_hist_train/len(train_dl):.4f}")

    # Test
    pred = model(x_test.float())[:, 0]
    loss = loss_fn(pred, y_test)
    logging.info(f'Test MSE: {loss.item():.4f}')
    logging.info(f'Test MAE: {torch.nn.L1Loss()(pred, y_test).item():.4f}')

if __name__ == '__main__':
    main()
