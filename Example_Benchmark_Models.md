PatchTST
PatchTST: Efficient Transformer model for multivariate forecasting using patched time series and channel-independence for scalable long-term predictions.

The PatchTST model is an efficient Transformer-based model for multivariate time series forecasting.
It is based on two key components: - segmentation of time series into windows (patches) which are served as input tokens to Transformer - channel-independence. where each channel contains a single univariate time series.
References
Nie, Y., Nguyen, N. H., Sinthong, P., & Kalagnanam, J. (2022). “A Time Series is Worth 64 Words: Long-term Forecasting with Transformers”
Figure 1. PatchTST.
Figure 1. PatchTST.
​
1. PatchTST
​
PatchTST
PatchTST(
    h,
    input_size,
    stat_exog_list=None,
    hist_exog_list=None,
    futr_exog_list=None,
    exclude_insample_y=False,
    encoder_layers=3,
    n_heads=16,
    hidden_size=128,
    linear_hidden_size=256,
    dropout=0.2,
    fc_dropout=0.2,
    head_dropout=0.0,
    attn_dropout=0.0,
    patch_len=16,
    stride=8,
    revin=True,
    revin_affine=False,
    revin_subtract_last=True,
    activation="gelu",
    res_attention=True,
    batch_normalization=False,
    learn_pos_embed=True,
    loss=MAE(),
    valid_loss=None,
    max_steps=5000,
    learning_rate=0.0001,
    num_lr_decays=-1,
    early_stop_patience_steps=-1,
    val_monitor="ptl/val_loss",
    val_check_steps=100,
    batch_size=32,
    valid_batch_size=None,
    windows_batch_size=1024,
    inference_windows_batch_size=1024,
    start_padding_enabled=False,
    training_data_availability_threshold=0.0,
    step_size=1,
    scaler_type="identity",
    random_seed=1,
    drop_last_loader=False,
    alias=None,
    optimizer=None,
    optimizer_kwargs=None,
    lr_scheduler=None,
    lr_scheduler_kwargs=None,
    dataloader_kwargs=None,
    **trainer_kwargs
)
Bases: BaseModel
PatchTST
The PatchTST model is an efficient Transformer-based model for multivariate time series forecasting.
It is based on two key components:
segmentation of time series into windows (patches) which are served as input tokens to Transformer
channel-independence, where each channel contains a single univariate time series.
Parameters:
Name	Type	Description	Default
h	int	forecast horizon.	required
input_size	int	autorregresive inputs size, y=[1,2,3,4] input_size=2 -> y_[t-2:t]=[1,2].	required
stat_exog_list	str list	static exogenous columns.	None
hist_exog_list	str list	historic exogenous columns.	None
futr_exog_list	str list	future exogenous columns.	None
exclude_insample_y	bool	the model skips the autoregressive features y[t-input_size:t] if True.	False
encoder_layers	int	number of layers for encoder.	3
n_heads	int	number of multi-head’s attention.	16
hidden_size	int	units of embeddings and encoders.	128
linear_hidden_size	int	units of linear layer.	256
dropout	float	dropout rate for residual connection.	0.2
fc_dropout	float	dropout rate for linear layer.	0.2
head_dropout	float	dropout rate for Flatten head layer.	0.0
attn_dropout	float	dropout rate for attention layer.	0.0
patch_len	int	length of patch. Note: patch_len = min(patch_len, input_size + stride).	16
stride	int	stride of patch.	8
revin	bool	bool to use RevIn.	True
revin_affine	bool	bool to use affine in RevIn.	False
revin_subtract_last	bool	bool to use substract last in RevIn.	True
activation	str	activation from [‘gelu’,‘relu’].	‘gelu’
res_attention	bool	bool to use residual attention.	True
batch_normalization	bool	bool to use batch normalization.	False
learn_pos_embed	bool	bool to learn positional embedding.	True
loss	PyTorch module	instantiated train loss class from losses collection.	MAE()
valid_loss	PyTorch module	instantiated valid loss class from losses collection.	None
max_steps	int	maximum number of training steps.	5000
learning_rate	float	learning rate between (0, 1).	0.0001
num_lr_decays	int	number of learning rate decays, evenly distributed across max_steps.	-1
early_stop_patience_steps	int	number of validation iterations before early stopping.	-1
val_monitor	str	metric to monitor for early stopping. Valid options: “ptl/val_loss”, “valid_loss”, “train_loss”. Default: “ptl/val_loss”.	‘ptl/val_loss’
val_check_steps	int	number of training steps between every validation loss check.	100
batch_size	int	number of different series in each batch.	32
valid_batch_size	int	number of different series in each validation and test batch, if None uses batch_size.	None
windows_batch_size	int	number of windows to sample in each training batch, default uses all.	1024
inference_windows_batch_size	int	number of windows to sample in each inference batch.	1024
start_padding_enabled	bool	if True, the model will pad the time series with zeros at the beginning, by input size.	False
training_data_availability_threshold	Union[float, List[float]]	minimum fraction of valid data points required for training windows. Single float applies to both insample and outsample; list of two floats specifies [insample_fraction, outsample_fraction]. Default 0.0 allows windows with only 1 valid data point (current behavior).	0.0
step_size	int	step size between each window of temporal data.	1
scaler_type	str	type of scaler for temporal inputs normalization see temporal scalers.	‘identity’
random_seed	int	random_seed for pytorch initializer and numpy generators.	1
drop_last_loader	bool	if True TimeSeriesDataLoader drops last non-full batch.	False
alias	str	optional, Custom name of the model.	None
optimizer	Subclass of ‘torch.optim.Optimizer’	optional, user specified optimizer instead of the default choice (Adam).	None
optimizer_kwargs	dict	optional, list of parameters used by the user specified optimizer.	None
lr_scheduler	Subclass of ‘torch.optim.lr_scheduler.LRScheduler’	optional, user specified lr_scheduler instead of the default choice (StepLR).	None
lr_scheduler_kwargs	dict	optional, list of parameters used by the user specified lr_scheduler.	None
dataloader_kwargs	dict	optional, list of parameters passed into the PyTorch Lightning dataloader by the TimeSeriesDataLoader.	None
**trainer_kwargs	int	keyword trainer arguments inherited from PyTorch Lighning’s trainer.	
​
PatchTST.fit
fit(
    dataset, val_size=0, test_size=0, random_seed=None, distributed_config=None
)
Fit.
The fit method, optimizes the neural network’s weights using the initialization parameters (learning_rate, windows_batch_size, …) and the loss function as defined during the initialization. Within fit we use a PyTorch Lightning Trainer that inherits the initialization’s self.trainer_kwargs, to customize its inputs, see PL’s trainer arguments.
The method is designed to be compatible with SKLearn-like classes and in particular to be compatible with the StatsForecast library.
By default the model is not saving training checkpoints to protect disk memory, to get them change enable_checkpointing=True in __init__.
Parameters:
Name	Type	Description	Default
dataset	TimeSeriesDataset	NeuralForecast’s TimeSeriesDataset, see documentation.	required
val_size	int	Validation size for temporal cross-validation.	0
random_seed	int	Random seed for pytorch initializer and numpy generators, overwrites model.init’s.	None
test_size	int	Test size for temporal cross-validation.	0
Returns:
Type	Description
None	
​
PatchTST.predict
predict(
    dataset,
    test_size=None,
    step_size=1,
    random_seed=None,
    quantiles=None,
    h=None,
    explainer_config=None,
    **data_module_kwargs
)
Predict.
Neural network prediction with PL’s Trainer execution of predict_step.
Parameters:
Name	Type	Description	Default
dataset	TimeSeriesDataset	NeuralForecast’s TimeSeriesDataset, see documentation.	required
test_size	int	Test size for temporal cross-validation.	None
step_size	int	Step size between each window.	1
random_seed	int	Random seed for pytorch initializer and numpy generators, overwrites model.init’s.	None
quantiles	list	Target quantiles to predict.	None
h	int	Prediction horizon, if None, uses the model’s fitted horizon. Defaults to None.	None
explainer_config	dict	configuration for explanations.	None
**data_module_kwargs	dict	PL’s TimeSeriesDataModule args, see documentation.	
Returns:
Type	Description
None	
​
Usage example
import pandas as pd
import matplotlib.pyplot as plt

from neuralforecast import NeuralForecast
from neuralforecast.models import PatchTST
from neuralforecast.losses.pytorch import DistributionLoss
from neuralforecast.utils import AirPassengersPanel, AirPassengersStatic, augment_calendar_df

AirPassengersPanel, calendar_cols = augment_calendar_df(df=AirPassengersPanel, freq='M')

Y_train_df = AirPassengersPanel[AirPassengersPanel.ds<AirPassengersPanel['ds'].values[-12]] # 132 train
Y_test_df = AirPassengersPanel[AirPassengersPanel.ds>=AirPassengersPanel['ds'].values[-12]].reset_index(drop=True) # 12 test

model = PatchTST(h=12,
                 input_size=104,
                 patch_len=24,
                 stride=24,
                 revin=False,
                 hidden_size=16,
                 n_heads=4,
                 scaler_type='robust',
                 loss=DistributionLoss(distribution='StudentT', level=[80, 90]),
                 learning_rate=1e-3,
                 max_steps=500,
                 val_check_steps=50,
                 early_stop_patience_steps=2)

nf = NeuralForecast(
    models=[model],
    freq='ME'
)
nf.fit(df=Y_train_df, static_df=AirPassengersStatic, val_size=12)
forecasts = nf.predict(futr_df=Y_test_df)

Y_hat_df = forecasts.reset_index(drop=False).drop(columns=['unique_id','ds'])
plot_df = pd.concat([Y_test_df, Y_hat_df], axis=1)
plot_df = pd.concat([Y_train_df, plot_df])

if model.loss.is_distribution_output:
    plot_df = plot_df[plot_df.unique_id=='Airline1'].drop('unique_id', axis=1)
    plt.plot(plot_df['ds'], plot_df['y'], c='black', label='True')
    plt.plot(plot_df['ds'], plot_df['PatchTST-median'], c='blue', label='median')
    plt.fill_between(x=plot_df['ds'][-12:], 
                    y1=plot_df['PatchTST-lo-90'][-12:].values, 
                    y2=plot_df['PatchTST-hi-90'][-12:].values,
                    alpha=0.4, label='level 90')
    plt.grid()
    plt.legend()
    plt.plot()
else:
    plot_df = plot_df[plot_df.unique_id=='Airline1'].drop('unique_id', axis=1)
    plt.plot(plot_df['ds'], plot_df['y'], c='black', label='True')
    plt.plot(plot_df['ds'], plot_df['PatchTST'], c='blue', label='Forecast')
    plt.legend()
    plt.grid()



# TimesFM 2.5

sage example
Copied
import numpy as np
import torch

from transformers import TimesFm2_5ModelForPrediction


model = TimesFm2_5ModelForPrediction.from_pretrained(
    "google/timesfm-2.5-200m-transformers",
    device_map="auto",
)

forecast_input = [
    np.sin(np.linspace(0, 20, 100)),
    np.sin(np.linspace(0, 20, 200)),
    np.sin(np.linspace(0, 20, 400)),
]
forecast_input_tensor = [torch.tensor(ts, dtype=torch.float32, device=model.device) for ts in forecast_input]

with torch.no_grad():
    outputs = model(past_values=forecast_input_tensor, return_dict=True)
    point_forecast = outputs.mean_predictions
    quantile_forecast = outputs.full_predictions

# Chronos-2

Install the package

pip install "chronos-forecasting>=2.0"

Make zero-shot predictions using the pandas API

import pandas as pd  # requires: pip install 'pandas[pyarrow]'
from chronos import Chronos2Pipeline

pipeline = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cuda")

# Load historical target values and past values of covariates
context_df = pd.read_parquet("https://autogluon.s3.amazonaws.com/datasets/timeseries/electricity_price/train.parquet")

# (Optional) Load future values of covariates
future_df = pd.read_parquet("https://autogluon.s3.amazonaws.com/datasets/timeseries/electricity_price/test.parquet").drop(columns="target")

# Generate predictions with covariates
pred_df = pipeline.predict_df(
    context_df,
    future_df=future_df,
    prediction_length=24,  # Number of steps to forecast
    quantile_levels=[0.1, 0.5, 0.9],  # Quantiles for probabilistic forecast
    id_column="id",  # Column identifying different time series
    timestamp_column="timestamp",  # Column with datetime information
    target="target",  # Column(s) with time series values to predict
)