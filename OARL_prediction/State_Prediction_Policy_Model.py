import torch
import torch.nn as nn
import numpy as np
import math
from torch.autograd import Variable
from tensorflow.python.keras.layers import LSTM, Dense, Input
from tensorflow.python.keras.models import Model, load_model, Sequential
from robust_ddpg import RobustDeterministicActorCriticNet

class PredictionWithPolicy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PredictionWithPolicy, self).__init__()
        self.flat = nn.Flatten()
        self.linear = nn.Sequential(
            nn.Linear(state_dim + action_dim, 64),
            #nn.ReLU(),
            nn.Linear(64, 32),
            #nn.ReLU(),
            nn.Linear(32, 16),
            #nn.ReLU(),
            nn.Linear(16, state_dim)
            #nn.ReLU()
        )

    def forward(self, x):
        #print(x)
        x = self.flat(x)
        #x.flatten()
        #x = Variable(torch.from_numpy(x), requires_grad=True)
        logits = self.linear(x)
        return logits


class LstmNStepNet(nn.Module):
    def __init__(self, state_dim, action_dim, t=10, n=15, pdt=0.67):
        super(LstmNStepNet, self).__init__()
        """
        Architecture:
        [t tuples of (gt state, policy action) into LSTM cell,
        Another LSTM cell,
        Another LSTM cell,
        Dense cell to predict next state]
        Add loss from prediction
        Get policy action
        If pred < 10: append tuple of (gt state, policy action) to inputs
        else: append tuple of (pred state, policy action) to inputs
        Get next output
        Add loss
        Repeat for 15 predictions
        Update parameters based on loss from predicted states, not 10 "warmup predictions"
        """
        self.t = t
        self.n = n
        self.pdt = pdt
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.LSTM(input_size=state_dim+action_dim, hidden_size=32, num_layers=3),
            nn.Linear(32, 16),
            nn.Linear(16, state_dim)
        )

    def forward(self, x):
        return self.network(x)


class nStepTrainer:
    def __init__(self, state_dim, action_dim, dataset, action_model, t=10, n=15, pdt=0.67, batch_size=5000, epochs=500):
        """
        Goal: input t timesteps, output n timesteps
        We will follow 67% PDT training steps from Chiappa et al 2017
        Input 10 state/actions, output 15 states (using actions from policy for next input)
        First 10 prediction are based off gt-observation, next 5 are from predicted observation
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.policy = action_model
        self.data = dataset
        self.model = LstmNStepNet(state_dim, action_dim, t, n, pdt)
        self.t = t
        self.n = n
        self.pdt = pdt
        self.batch_size = 5000
        self.epochs = epochs
        self.train_split = .8
        self.val_split = .1

    def train(self, save=False, fname=None):
        inputs, outputs = self.partition_data()
        inputs, outputs = self.batch_data(inputs, outputs)
        inX, inY, valX, valY, testX, testY = self.train_val_split(inputs, outputs)



    def partition_data(self):
        states = []
        actions = []
        next_states = []
        for i, data in enumerate(self.data):
            if not data[3]:
                states.append(data[0])
                actions.append(data[1])
                next_states.append(data[2])

        states = np.asarray(states)
        actions = np.asarray(actions)
        next_states = np.asarray(next_states)

        inputs = np.concatenate([states, actions], axis=1)
        outputs = next_states

        return inputs, outputs

    def batch_data(self, inputs, outputs):
        in_batches = []
        out_batches = []
        for i in range(len(inputs) // self.batch_size):
            in_batches.append(inputs[(self.batch_size * i):self.batch_size * (i + 1)])
            out_batches.append(outputs[(self.batch_size * i):self.batch_size * (i + 1)])

        in_batches.append(inputs[self.batch_size * (len(inputs) // self.batch_size):])
        out_batches.append(outputs[self.batch_size * (len(inputs) // self.batch_size):])

        return np.asarray(in_batches), np.asarray(out_batches)

    def train_val_split(self, inputs, outputs):
        train_stop = math.floor(len(inputs) * self.train_split)
        val_stop = math.floor(len(inputs) * (self.train_split + self.val_split))
        inX = inputs[:train_stop]
        inY = outputs[:train_stop]
        valX = inputs[train_stop:val_stop]
        valY = outputs[train_stop:val_stop]
        testX = inputs[val_stop:]
        testY = outputs[val_stop:]

        return inX, inY, valX, valY, testX, testY





#######################################################################################################################



class PredictionTrainer:
    def __init__(self, state_dim, action_dim, dataset, action_model=None, det=True, batch_size=5000, epochs=500):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.policy = action_model
        self.data = dataset
        self.model = PredictionWithPolicy(state_dim, action_dim)
        self.batch_size = batch_size
        self.epochs = epochs
        self.train_split = .8
        self.val_split = .1
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.policy_temperature = .05

        if self.policy is None:
            self.pol_reg = False
        else:
            self.pol_reg = True

        self.deterministic = det

    def train(self, save=False, fname=None):
        inputs, outputs = self.partition_data()
        inputs, outputs = self.batch_data(inputs, outputs)
        inX, inY, valX, valY, testX, testY = self.train_val_split(inputs, outputs)
        for epoch in range(self.epochs):
            train_loss = 0
            val_loss = 0
            train_mae = 0
            val_mae = 0
            for i in range(len(inX)):
                x_batch = torch.from_numpy(inX[i]).float()
                y_batch = torch.from_numpy(inY[i]).float()

                pred_batch = self.model(x_batch)

                if self.pol_reg:
                    if self.deterministic:
                        kl_loss = self.determine_det_policy_loss(pred_batch, y_batch)
                    else:
                        kl_loss = self.determine_stoch_policy_loss(pred_batch, y_batch)


                self.optimizer.zero_grad()

                loss = []
                mae = []
                for j in range(len(pred_batch)):
                    loss.append(torch.mean(torch.square(pred_batch[j] - y_batch[j]))) #mse loss
                    mae.append(torch.mean(torch.abs(pred_batch[j] - y_batch[j])))

                if self.pol_reg:
                    loss = (sum(loss) / len(loss)) + (self.policy_temperature * kl_loss)
                else:
                    loss = (sum(loss) / len(loss))
                train_mae += (sum(mae) / len(mae))
                train_loss += loss

                loss.backward() #Should i be computing the loss more often?
                self.optimizer.step()

            train_mae = train_mae / len(inX)
            train_loss = train_loss / len(inX)

            for i in range(len(valX)):
                with torch.no_grad():
                    x_batch = torch.from_numpy(valX[i]).float()
                    y_batch = torch.from_numpy(valY[i]).float()

                    pred_batch = self.model(x_batch)

                    if self.pol_reg:
                        if self.deterministic:
                            kl_loss = self.determine_det_policy_loss(pred_batch, y_batch)
                        else:
                            kl_loss = self.determine_stoch_policy_loss(pred_batch, y_batch)

                    loss = []
                    mae = []
                    for j in range(len(pred_batch)):
                        loss.append(torch.mean(torch.square(pred_batch[j] - y_batch[j])))
                        mae.append(torch.mean(torch.abs(pred_batch[j] - y_batch[j])))

                    if self.pol_reg:
                        loss = (sum(loss) / len(loss)) + (self.policy_temperature * kl_loss)
                    else:
                        loss = (sum(loss) / len(loss))
                    val_mae += (sum(mae) / len(mae))
                    val_loss += loss

            val_mae = val_mae / len(valX)
            val_loss = val_loss / len(valX)


            if epoch % 10 == 0:
                print("Epoch: {}, Train Loss: {}, Train MAE: {}, Val Loss: {}, Val MAE: {}".format(epoch, train_loss, train_mae, val_loss, val_mae))

        if save:
            assert fname is not None, "Must specify filename"
            self.save_model(fname)

    def save_model(self, fname):
        torch.save(self.model.state_dict(), fname)

    def load_model(self, fname):
        self.model.load_state_dict(torch.load(fname))
        self.model.eval() #Set batch and normalize layers to evaluation mode


    def determine_stoch_policy_loss(self, y_pred, y_true):
        with torch.no_grad():
            policy_true = self.policy(y_true)
            policy_pred = self.policy(y_pred)

            loss = []
            for i in range(len(policy_pred)):
                sample_loss = 0
                for j in range(self.action_dim):
                    log_term = torch.log(policy_pred[i][j] / policy_true[i][j])
                    sample_loss += policy_pred[i][j] * log_term
                loss.append(sample_loss)
                #print(sample_loss)
            batch_loss = sum(loss) / len(loss)

        return batch_loss

    def determine_det_policy_loss(self, y_pred, y_true):
        with torch.no_grad():
            policy_true = self.policy(y_true)
            policy_pred = self.policy(y_pred)

            loss = []
            for i in range(len(policy_pred)):
                sample_loss = 0
                for j in range(self.action_dim):
                    sample_loss += torch.square(policy_true[i][j] - policy_pred[i][j])
                sample_loss = sample_loss / self.action_dim
                loss.append(sample_loss)
                #print(sample_loss)
            batch_loss = sum(loss) / len(loss)

        return batch_loss


    def partition_data(self):
        states = []
        actions = []
        next_states = []
        for i, data in enumerate(self.data):
            if not data[3]:
                states.append(data[0])
                actions.append(data[1])
                next_states.append(data[2])

        states = np.asarray(states)
        actions = np.asarray(actions)
        next_states = np.asarray(next_states)

        inputs = np.concatenate([states, actions], axis=1)
        outputs = next_states

        return inputs, outputs

    def batch_data(self, inputs, outputs):
        in_batches = []
        out_batches = []
        for i in range(len(inputs) // self.batch_size):
            in_batches.append(inputs[(self.batch_size*i):self.batch_size*(i+1)])
            out_batches.append(outputs[(self.batch_size * i):self.batch_size * (i + 1)])

        in_batches.append(inputs[self.batch_size * (len(inputs) // self.batch_size):])
        out_batches.append(outputs[self.batch_size * (len(inputs) // self.batch_size):])

        return np.asarray(in_batches), np.asarray(out_batches)

    def train_val_split(self, inputs, outputs):
        train_stop = math.floor(len(inputs) * self.train_split)
        val_stop = math.floor(len(inputs) * (self.train_split + self.val_split))
        inX = inputs[:train_stop]
        inY = outputs[:train_stop]
        valX = inputs[train_stop:val_stop]
        valY = outputs[train_stop:val_stop]
        testX = inputs[val_stop:]
        testY = outputs[val_stop:]

        return inX, inY, valX, valY, testX, testY


def train_pendulum(config, policy_loss):
    data = np.load('transitions/Pendulum_DDPG_transition.npy', allow_pickle=True)
    if policy_loss:
        policy_path = "models/vanilla-ddpg/ddpg_InvertedPendulum-v2/model_best"  # Network is loaded from config.network_fn()

        state_dict = torch.load('%s.model' % policy_path)
        model = RobustDeterministicActorCriticNet(
            config.state_dim, config.action_dim,
            config.actor_network, config.critic_network,
            config.mini_batch_size,
            actor_opt_fn=lambda params: torch.optim.Adam(params, lr=0.0),
            critic_opt_fn=lambda params: torch.optim.Adam(params, lr=0.0),
            robust_params=config.certify_params)

        model.load_state_dict(state_dict)

        trainer = PredictionTrainer(4, 1, data, action_model=model)
    else:
        trainer = PredictionTrainer(4, 1, data)
    trainer.train(save=True, fname="OARL_prediction/Pendulum/Pendulum_dense_torch_pol_2.model")




