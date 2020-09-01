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
