from models.st_gat import ST_GAT
from tqdm import tqdm
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
import tensorflow as tf
import datetime
from utils.math_utils import un_z_score,RMSE,MAE,MAPE
import os
from spektral.data import BatchLoader
import numpy as np
from matplotlib import pyplot as plt

current_time = datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S")
# Create a folder with the datetime stamp
log_dir = f"./runs/{current_time}"
os.makedirs(log_dir)

train_log_dir = './runs/'+log_dir +'/train'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
eval_log_dir = './runs/'+log_dir +'/eval'
eval_summary_writer=tf.summary.create_file_writer(eval_log_dir)


def eval(model,dataloader,dataset_mean,dataset_std,type,epoch):
    """
    Evaluation function to evaluate model on data
    :param model Model to evaluate
    :param device Device to evaluate on
    :param dataloader Data loader
    :param type Name of evaluation type, e.g. Train/Val/Test
"""
    mae = 0
    rmse = 0
    mape = 0
    n = 0

    # Evaluate model on all data
    for i, batch in enumerate(dataloader):
        inputs, target = batch
        x,a=inputs
        print(x.shape,a.shape)
        if x.shape[0] == 1:
            print('PASS')
        elif x.shape[0]==50:
            x = x.astype(np.float32)
            a = a.astype(np.float32)
            inputs=(x,a)
            pred = model(inputs, training=False)
            # truth = batch.y.view(pred.shape)
            print("pred :",pred.shape)
            truth = np.reshape(target, (pred.shape[0], pred.shape[1]))
            if i == 0:
                y_pred = np.zeros((dataloader.steps_per_epoch, pred.shape[0], pred.shape[1]))
                y_truth = np.zeros((dataloader.steps_per_epoch, pred.shape[0], pred.shape[1]))
            truth = un_z_score(truth, dataset_mean, dataset_std)
            pred = un_z_score(pred, dataset_mean, dataset_std)
            y_pred[i, :pred.shape[0], :] = pred
            y_truth[i, :pred.shape[0], :] = truth
            rmse += RMSE(truth, pred)
            mae += MAE(truth, pred)
            mape += MAPE(truth, pred)
            n += 1
        else:
            pass
    rmse, mae, mape = rmse / n, mae / n, mape / n

    print(f'{type} {epoch}, MAE: {mae}, RMSE: {rmse}, MAPE: {mape}')

    #get the average score for each metric in each batch
    return rmse, mae, mape, y_pred, y_truth



@tf.function()
def train_step(model,batch,loss_fn,optimizer):
    with tf.GradientTape() as tape:
        inputs,target=batch
        x,a=inputs
        # print(x.shape,a.shape,target.shape)
        predictions = model(inputs, training=True)
        # print("predictions : ",predictions.shape)
        target = tf.reshape(target, (predictions.shape[0], predictions.shape[1]))
#         target=np.reshape(target,11400,9)
#         print(target.shape,type(target))
        loss = loss_fn(target, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

def model_train(train, val, config,dataset_mean,dataset_std):
    """
    Train the ST-GAT model. Evaluate on validation dataset as you go.
    :param train_dataloader Data loader of training dataset
    :param val_dataloader Dataloader of val dataset
    :param config configuration to use
    :param device Device to evaluate on
    """

    # Make the model. Each datapoint in the graph is 228x12: N x F (N = # nodes, F = time window)
    model = ST_GAT(gat_features=config['N_HIST'], out_channels=config['N_PRED'], n_nodes=config['N_NODE'],
                   dropout=config['DROPOUT'])
    optimizer = Adam(lr=config['INITIAL_LR'],decay=config['WEIGHT_DECAY'])
    loss_fn=MeanSquaredError()
    step=0
    train_dataloader = BatchLoader(train, batch_size=50,epochs=1)

    for epoch in range(1,config['EPOCHS']+1):
        for batch in train_dataloader:
            step+=1
            inputs, target = batch
            x, a = inputs
            if x.shape == (50, 228, 12):
                loss = train_step(model,batch,loss_fn,optimizer)
                with train_summary_writer.as_default():
                    tf.summary.scalar('train loss', loss, step=step)
                if step %50==0:
                    print(f"epoch : {epoch} , step {step} : loss {loss}")
            else:
                print('rejecting batch')

        if epoch % 5 ==0:
            print('eval mode')
            train_dataloader = BatchLoader(train, batch_size=50, epochs=1)
            train_mae, train_rmse, train_mape, _, _=eval(model,train_dataloader,dataset_mean,dataset_std,'Train',epoch)
            with train_summary_writer.as_default():
                tf.summary.scalar('train_mae', train_mae, step=step)
                tf.summary.scalar('train_rmse', train_rmse, step=step)
                tf.summary.scalar('train_mape', train_mape, step=step)


            val_dataloader = BatchLoader(val, batch_size=50, epochs=1)
            val_mae, val_rmse, val_mape, _, _ = eval(model, val_dataloader, dataset_mean, dataset_std, 'Val',epoch)
            with eval_summary_writer.as_default():
                tf.summary.scalar('val_mae', val_mae, step=step)
                tf.summary.scalar('val_rmse', val_rmse, step=step)
                tf.summary.scalar('val_mape', val_mape, step=step)


        print(f"epoch : {epoch} , step {step} : loss {loss}")

    return model

def model_test(model, test, dataset_mean, dataset_std, config):
    """
    Test the ST-GAT model
    :param test_dataloader Data loader of test dataset
    :param device Device to evaluate on
    """
    test_dataloader = BatchLoader(test, batch_size=50, epochs=1)
    _, _, _, y_pred, y_truth = eval(model,test_dataloader, dataset_mean, dataset_std,'Test',1)
    plot_prediction(test_dataloader, y_pred, y_truth, 0, config)

def plot_prediction(test_dataloader, y_pred, y_truth, node, config):
    # Calculate the truth
    s = y_truth.shape
    y_truth = y_truth.reshape(s[0], config['BATCH_SIZE'], config['N_NODE'], s[-1])
    # just get the first prediction out for the nth node
    y_truth = y_truth[:, :, node, 0]
    # Flatten to get the predictions for entire test dataset
    # y_truth = torch.flatten(y_truth)
    y_truth=tf.reshape(y_truth, shape=[-1])
    day0_truth = y_truth[:config['N_SLOT']]

    # Calculate the predicted
    s = y_pred.shape
    y_pred = y_pred.reshape(s[0], config['BATCH_SIZE'], config['N_NODE'], s[-1])
    # just get the first prediction out for the nth node
    y_pred = y_pred[:, :, node, 0]
    # Flatten to get the predictions for entire test dataset
    # y_pred = torch.flatten(y_pred)
    y_pred = tf.reshape(y_pred, shape=[-1])
    # Just grab the first day
    day0_pred = y_pred[:config['N_SLOT']]
    t = [t for t in range(0, config['N_SLOT'] * 5, 5)]
    plt.plot(t, day0_pred, label='ST-GAT')
    plt.plot(t, day0_truth, label='truth')
    plt.xlabel('Time (minutes)')
    plt.ylabel('Speed prediction')
    plt.title('Predictions of traffic over time')
    plt.legend()
    plt.savefig('predicted_times.png')
    plt.show()

# def load_from_checkpoint(checkpoint_path, config):
#     """
#     Load a model from the checkpoint
#     :param checkpoint_path Path to checkpoint
#     :param config Configuration to load model with
#     """
#     model = ST_GAT(in_channels=config['N_HIST'], out_channels=config['N_PRED'], n_nodes=config['N_NODE'])
#     checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
#     model.load_state_dict(checkpoint['model_state_dict'])
#
#     return model




