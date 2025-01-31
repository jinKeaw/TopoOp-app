from tbparse import SummaryReader

log_dir = r"C:\Users\Jin\PycharmProjects\DRL_TopoOp\src\GUI\Dashboard_Data\matricslog.tfevents"
reader = SummaryReader(log_dir)
df = reader.scalars
#print(df)

# Get all the unique tags from the 'tag' column
unique_tags = df['tag'].unique()

# Display the unique tags
#print(unique_tags)

import pandas as pd
import matplotlib.pyplot as plt

# to be implemented
## already implemented
### don't have the exact log file, doesn't mean you can't plot something related to it

###fig_metric_cumulative_reward = create_figure_metric_cumulative_reward()
##rollout/ep_len_mean
ep_len_mean_df = df[df['tag'] == 'rollout/ep_len_mean']
#print(ep_len_mean_df)
# Extract the 'step' and 'value' columns into variables
steps_ep_len_mean = ep_len_mean_df['step']
ep_len_mean = ep_len_mean_df['value']

##rollout/ep_rew_mean
ep_rew_mean_df = df[df['tag'] == 'rollout/ep_rew_mean']
#print(ep_rew_mean_df)
# Extract the 'step' and 'value' columns into variables
steps_rew_mean = ep_rew_mean_df['step']
ep_rew_mean = ep_rew_mean_df['value']


#time/fps = 0 for all timesteps
fps_df = df[df['tag'] == 'time/fps']
#print(fps_df)
# Extract the 'step' and 'value' columns into variables
steps_fps = fps_df['step']
fps = fps_df['value']


#time_elapsed
time_elapsed_df = df[df['tag'] == 'time_elapsed']
#print(time_elapsed_df)
# Extract the 'step' and 'value' columns into variables
steps_time_elapsed = time_elapsed_df['step']
time_elapsed = time_elapsed_df['value']

#total_timesteps
total_timesteps_df = df[df['tag'] == 'total_timesteps']
#print(total_timesteps_df)
# Extract the 'step' and 'value' columns into variables
steps_total_timesteps = total_timesteps_df['step']
total_timesteps = total_timesteps_df['value']

#train/approx_kl
approx_kl_df = df[df['tag'] == 'train/approx_kl']
#print(approx_kl_df)
# Extract the 'step' and 'value' columns into variables
steps_approx_kl = approx_kl_df['step']
approx_kl= approx_kl_df['value']

#train/clip_fraction
clip_fraction_df = df[df['tag'] == 'train/clip_fraction']
#print(clip_fraction_df)
# Extract the 'step' and 'value' columns into variables
steps_clip_fraction = clip_fraction_df['step']
clip_fraction= clip_fraction_df['value']

#train/clip_range
clip_range_df = df[df['tag'] == 'train/clip_range']
#print(clip_range_df)
# Extract the 'step' and 'value' columns into variables
steps_clip_range = clip_range_df['step']
clip_range= clip_range_df['value']

##train/entropy_loss
entropy_loss_df = df[df['tag'] == 'train/entropy_loss']
#print(entropy_loss_df)
# Extract the 'step' and 'value' columns into variables
steps_entropy_loss = entropy_loss_df['step']
entropy_loss= entropy_loss_df['value']

#train/explained_variance
explained_variance_df = df[df['tag'] == 'train/explained_variance']
#print(explained_variance_df)
# Extract the 'step' and 'value' columns into variables
steps_explained_variance = explained_variance_df['step']
explained_variance= explained_variance_df['value']

##train/learning_rate
learning_rate_df = df[df['tag'] == 'train/learning_rate']
#print(learning_rate_df)
# Extract the 'step' and 'value' columns into variables
steps_learning_rate= learning_rate_df['step']
learning_rate= learning_rate_df['value']

#train/loss
loss_df = df[df['tag'] == 'train/loss']
#print(loss_df)
# Extract the 'step' and 'value' columns into variables
steps_loss= loss_df['step']
loss= loss_df['value']

###fig_metric_value_estimate = create_figure_metric_value_estimate()

##train/policy_gradient_loss
policy_gradient_loss_df = df[df['tag'] == 'train/policy_gradient_loss']
#print(policy_gradient_loss_df)
# Extract the 'step' and 'value' columns into variables
steps_policy_gradient_loss= policy_gradient_loss_df['step']
policy_gradient_loss= policy_gradient_loss_df['value']

##train/value_loss all value = 0
value_loss_df = df[df['tag'] == 'train/value_loss']
#print(value_loss_df)
# Extract the 'step' and 'value' columns into variables
steps_value_loss = value_loss_df['step']
value_loss = value_loss_df['value']



