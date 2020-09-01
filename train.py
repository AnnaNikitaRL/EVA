def train(Q_param, Q_target, replay, value_buffer, config):
    """
    performs loop for a train step for EVA
    """
    def choose_action_embedding(state, n_actions, lambd, epsilon):
        """
        EVA policy is dictated by the action-value function 
        
        Q(s, a) = lambda * Q_param(s, a) + (1 - lambda) * QNP(s, a)
        
        we choose action based on the argmax of this Q value function.
        With probablity epsilon action is taken randomly
        
        Input: state 
        return: action, embedding for state
        """
        Q_param, embedding = qnet(torch.FloatTensor(state).to(device).unsqueeze(0))
        Q_param            = Q_param.detach().cpu().squeeze()
        embedding          = embedding.detach().cpu().squeeze().numpy()
        
        # Chooses random action with probability epsilon
        if np.random.random() < epsilon:
            action = np.random.randint(n_actions)
        else:
            if len(value_buffer) == value_buffer.capacity:
                Q_param = lambd * Q_param + (1-lambd) * value_buffer.nn_Q(embedding)
            action = torch.argmax(Q_param).item()
        return action, embedding

    def step(action):
        """returns next state and revard based on action"""
        next_state, reward, is_terminal,_ = env.step(action)
        if is_terminal:
            next_state = None
        return next_state, reward, is_terminal

    def train_step(config):
        """
        train step per batch, single step of the optimisation as in DQN model.
        Implementation is inspired by official pytorch turorial for DQN
        https://pytorch.org/tutorials/
        """
        # Take state, action, reward, next_state per batch from replay buffer
        state_batch, action_batch, reward_batch, next_state_batch = zip(*replay.sample(config.batch_size))
        state_batch      = torch.from_numpy(np.stack(state_batch)).to(device)
        action_batch     = torch.LongTensor(action_batch).view(-1,1).to(device)
        
        # Compute mask of non-terminal states
        non_final_mask   = torch.BoolTensor([next_state_batch_i is not None 
                                             for next_state_batch_i in next_state_batch]).view(-1).to(device)

        # concatenate non terminal state batch elements
        next_state_batch = torch.from_numpy(np.stack([next_state_batch_i 
                                                      for next_state_batch_i in next_state_batch 
                                                      if next_state_batch_i is not None])).to(device)

        reward_batch     = torch.FloatTensor(reward_batch).view(-1,1).to(device)

        # Clean optimizer
        optimizer.zero_grad()
        
        # Compute Q values for all next states. 
        # Expected values for non-terminal next states are computed based on older target net. 
        # The use of mask helps to have expected state value or 0 for terminal state
        Q_predicted              = qnet(state_batch)[0].gather(1, action_batch)
        Q_target                 = torch.zeros_like(Q_predicted)
        Q_target[non_final_mask] = target_net(next_state_batch)[0].max(1, keepdim=True)[0].detach()
        
        # Compute expected Q values and optimize the model
        loss = F.mse_loss(Q_predicted, Q_target*config.gamma + reward_batch)
        loss.backward()
        optimizer.step() 






    total_rewards     = []  
    for episode in range(1, config.n_episodes + 1, 1):
        
        if episode %  config.test_freq == 0:
            test(episode)    
            
        state               = env.reset()
        is_terminal         = False
        episode_reward      = 0
        
        for t in range(config.t_max):
            action, embedding  = choose_action_embedding( state, 
                                                          n_actions, 
                                                          epsilon(global_step), 
                                                          config.lambd )  
            next_state, reward, is_terminal = step(action)
            
            replay.push(state, action, reward, next_state, embedding)        
            state          = next_state
            episode_reward += reward 
            
            if len(replay) >= config.batch_size:           
                train_step(config.batch_size) 
            
            # we call trajectory central planning with TCP frequency and when replay is full
            if ((global_step % config.tcp_frequency) == 0) and  (len(replay) == replay.capacity) :           
                trajectory_central_planning(replay, 
                                            embedding, 
                                            value_buffer, 
                                            num_neighbors=config.num_tcp_paths, 
                                            path_length=config.path_length)            
            
            # update of target net periodically
            global_step +=1       
            if (global_step % config.t_update) == 0:
                target_net.load_state_dict(qnet.state_dict())   
                
            if is_terminal:
                total_rewards.append(episode_reward)
                break
