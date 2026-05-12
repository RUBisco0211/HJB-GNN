import jax.lax as lax
import jax.numpy as jnp
import jax.random as jr
import optax
import jax
import functools as ft
import jax.tree_util as jtu
import numpy as np
import einops as ei
from enum import Enum
import pickle
import os

from typing import Optional, Tuple, NamedTuple
from flax.training.train_state import TrainState
from jaxproxqp.jaxproxqp import JaxProxQP

from hjbgnn.utils.typing import Action, Params, PRNGKey, Array, State, AgentState
from hjbgnn.utils.graph import GraphsTuple
from hjbgnn.utils.utils import merge01, jax_vmap, mask2index, tree_merge
from hjbgnn.trainer.data import Rollout
from hjbgnn.trainer.buffer import MaskedReplayBuffer
from hjbgnn.trainer.utils import compute_norm_and_clip, jax2np, tree_copy, empty_grad_tx
from hjbgnn.env.base import MultiAgentEnv
from hjbgnn.algo.module.cbf import CBF
from hjbgnn.algo.module.policy import DeterministicPolicy, DeterministicValuePolicy
from .gcbf import GCBF


class Batch(NamedTuple):
    graph: GraphsTuple
    safe_mask: Array
    unsafe_mask: Array
    u_kkt: Action

class TrainingState(Enum):
    WARMUP = 0
    UPDATEVALUE = 1
    UPDATEACTIOR = 2

class HJBGNN(GCBF):

    def __init__(
            self,
            env: MultiAgentEnv,
            node_dim: int,
            edge_dim: int,
            state_dim: int,
            action_dim: int,
            n_agents: int,
            gnn_layers: int,
            batch_size: int,
            buffer_size: int,
            horizon: int = 32,
            lr_actor: float = 3e-5,
            lr_value: float = 3e-5,
            lr_cbf: float = 3e-5,
            alpha: float = 1.0,
            eps: float = 0.02,
            inner_epoch: int = 8,
            loss_value_coef: float = 1, 
            loss_value_coef2: float = 0.1,
            loss_action_coef: float = 0.001,
            loss_unsafe_coef: float = 1.,
            loss_safe_coef: float = 1.,
            loss_h_dot_coef: float = 0.2,
            max_grad_norm: float = 2.,
            seed: int = 0,
            **kwargs
    ):
        super(GCBF, self).__init__(
            env=env,
            node_dim=node_dim,
            edge_dim=edge_dim,
            action_dim=action_dim,
            n_agents=n_agents
        )

        # set hyperparameters
        self.batch_size = batch_size
        self.lr_actor = lr_actor
        self.lr_value = lr_value
        self.lr_cbf = lr_cbf
        self.alpha = alpha
        self.eps = eps
        self.inner_epoch = inner_epoch
        self.loss_action_coef = loss_action_coef
        self.loss_unsafe_coef = loss_unsafe_coef
        self.loss_safe_coef = loss_safe_coef
        self.loss_h_dot_coef = loss_h_dot_coef
        self.gnn_layers = gnn_layers
        self.max_grad_norm = max_grad_norm
        self.seed = seed
        self.horizon = horizon
        self.loss_value_coef = loss_value_coef
        self.loss_value_coef2 = loss_value_coef2
        # set nominal graph for initialization of the neural networks
        nominal_graph = GraphsTuple(
            nodes=jnp.zeros((n_agents, node_dim)),
            edges=jnp.zeros((n_agents, edge_dim)),
            states=jnp.zeros((n_agents*2, state_dim)),
            n_node=jnp.array(n_agents),
            n_edge=jnp.array(n_agents),
            senders=jnp.arange(n_agents),
            receivers=jnp.arange(n_agents),
            node_type=jnp.zeros((n_agents,)),
            env_states=jnp.zeros((n_agents,)),
        )
        self.nominal_graph = nominal_graph

        # set up CBF
        self.cbf = CBF(
            node_dim=node_dim,
            edge_dim=edge_dim,
            n_agents=n_agents,
            gnn_layers=gnn_layers
        )
        key = jr.PRNGKey(seed) 
        cbf_key, key = jr.split(key)
        cbf_params = self.cbf.net.init(cbf_key, nominal_graph, self.n_agents)
        self.cbf_lr_scheduler = optax.exponential_decay(
            init_value=lr_cbf,
            transition_steps=1000 * 16 * self.inner_epoch * int(self._env._max_step/self.batch_size),
            decay_rate=0.4
        )
        cbf_optim = optax.adamw(learning_rate=self.cbf_lr_scheduler, weight_decay=1e-3)
        self.cbf_optim = optax.apply_if_finite(cbf_optim, 1_000_000)
        self.cbf_train_state = TrainState.create(
            apply_fn=self.cbf.get_cbf,
            params=cbf_params,
            tx=self.cbf_optim
        )
        self.cbf_tgt = TrainState.create(
            apply_fn=self.cbf.get_cbf,
            params=tree_copy(cbf_params),
            tx=empty_grad_tx())

        # set up actor
        self.actor = DeterministicPolicy(
            node_dim=node_dim,
            edge_dim=edge_dim,
            action_dim=action_dim,
            n_agents=n_agents
        )
        actor_key, key = jr.split(key)
        actor_params = self.actor.net.init(actor_key, nominal_graph, self.n_agents)
        self.actor_lr_scheduler = optax.exponential_decay(
            init_value=lr_actor,
            transition_steps=500 * 16 * self.inner_epoch * int(self._env._max_step/self.batch_size),
            decay_rate=0.4
        )
        actor_optim = optax.adamw(learning_rate=self.actor_lr_scheduler, weight_decay=1e-3)
        self.actor_optim = optax.apply_if_finite(actor_optim, 1_000_000)
        self.actor_train_state = TrainState.create(
            apply_fn=self.actor.sample_action,
            params=actor_params,
            tx=self.actor_optim
        )

        # set up value
        self.value = DeterministicValuePolicy(
            node_dim=node_dim,
            edge_dim=edge_dim,
            action_dim=1,
            n_agents=n_agents,
            _n_state=4
        )
        value_key, key = jr.split(key)
        value_params = self.value.net.init(value_key, nominal_graph, self.n_agents)
        self.value_lr_scheduler = optax.exponential_decay(
            init_value=lr_value,
            transition_steps=500 * 16 * self.inner_epoch * int(self._env._max_step/self.batch_size),
            decay_rate=0.4
        )
        value_optim = optax.adamw(learning_rate=self.value_lr_scheduler, weight_decay=1e-3)
        self.value_optim = optax.apply_if_finite(value_optim, 1_000_000)
        self.value_train_state = TrainState.create(
            apply_fn=self.value.sample_value,
            params=value_params,
            tx=self.value_optim
        )
        
        # set up key
        self.key = key
        self.buffer = MaskedReplayBuffer(size=buffer_size)
        self.unsafe_buffer = MaskedReplayBuffer(size=buffer_size // 2)
        self.rng = np.random.default_rng(seed=seed + 1)

        self.training_state = TrainingState.WARMUP
        self.training_count = 0
        self.training_max = 8
        self.save_critc = 0
        self.save_actor_state = self.actor_train_state
        self.save_value_state = self.value_train_state

    @property
    def config(self) -> dict:
        return {
            'batch_size': self.batch_size,
            'lr_actor': self.lr_actor,
            'lr_cbf': self.lr_cbf,
            'alpha': self.alpha,
            'eps': self.eps,
            'inner_epoch': self.inner_epoch,
            'loss_action_coef': self.loss_action_coef,
            'loss_unsafe_coef': self.loss_unsafe_coef,
            'loss_safe_coef': self.loss_safe_coef,
            'loss_h_dot_coef': self.loss_h_dot_coef,
            'gnn_layers': self.gnn_layers,
            'seed': self.seed,
            'max_grad_norm': self.max_grad_norm,
            'horizon': self.horizon
        }
    @property
    def value_params(self) -> Params:
        return self.value_train_state.params
    
    @ft.partial(jax.jit, static_argnums=(0,))
    def safe_mask(self, unsafe_mask: Array) -> jnp.ndarray:
        # safe if in the horizon, the agent is always safe
        def safe_rollout(single_rollout_mask: Array) -> Array:
            safe_rollout_mask = jnp.ones_like(single_rollout_mask)
            for i in range(single_rollout_mask.shape[0]):
                start = 0 if i < self.horizon else i - self.horizon
                safe_rollout_mask = safe_rollout_mask.at[start: i + 1].set(
                    ((1 - single_rollout_mask[i]) * safe_rollout_mask[start: i + 1]).astype(jnp.bool_))
                # initial state is always safe
                safe_rollout_mask = safe_rollout_mask.at[0].set(1)
            return safe_rollout_mask

        safe = jax_vmap(jax_vmap(safe_rollout, in_axes=1, out_axes=1))(unsafe_mask)
        return safe

    def act(self, graph: GraphsTuple, params: Optional[Params] = None) -> Action:
        if params is None:
            params = self.actor_train_state.params
        action = self.actor.get_action(params, graph)
        lower_lim, upper_lim = self._env.action_lim()
        action = upper_lim[None,...]*action
        return action

    def val(self, graph: GraphsTuple, params: Optional[Params] = None) -> Action:
        if params is None:
            params = self.value_train_state.params
        value = self.value.get_value(params, graph)
        return value
    
    def step(self, graph: GraphsTuple, key: PRNGKey, params: Optional[Params] = None) -> Tuple[Action, Array]:
        if params is None:
            params = self.actor_params
        action, log_pi = self.actor_train_state.apply_fn(params, graph, key)
        lower_lim, upper_lim = self._env.action_lim()
        action = upper_lim[None,...]*action
        return action, log_pi

    @ft.partial(jax.jit, static_argnums=(0,), donate_argnums=1)
    def update_tgt(self, cbf_tgt: TrainState, cbf: TrainState, tau: float) -> TrainState:
        tgt_params = optax.incremental_update(cbf.params, cbf_tgt.params, tau)
        return cbf_tgt.replace(params=tgt_params)
    
    @ft.partial(jax.jit, static_argnums=(0,))
    def get_b_u_kkt(self, b_graph: GraphsTuple, cbfparams, actorparams, valueparams, num_step: int) -> Action: 
        b_u_kkt, V_x, lag_term, lagrange = jax_vmap(ft.partial(self.get_kkt_action, num_step=num_step, cbf_params=cbfparams, actor_params=actorparams, value_params=valueparams))(b_graph)
        return b_u_kkt, V_x, lag_term, lagrange

    @ft.partial(jax.jit, static_argnums=(0,))
    def get_batch_lagrange(self, b_graph: GraphsTuple, cbf_params, actor_params, value_params) : #-> Action 
        lagrange, V_x, V, h_x, h_dot_m, h, dyn_f, dyn_g = jax_vmap(ft.partial(self.get_single_lagrange, cbf_params=cbf_params, actor_params=actor_params, value_params=value_params))(b_graph)
        return lagrange, V_x, V, h_x, h_dot_m, h, dyn_f, dyn_g

    @ft.partial(jax.jit, static_argnums=(0,))
    def get_batch_h_gradient(self, b_graph: GraphsTuple, params) : #-> Action     
        batch_h, batch_h_x, batch_dyn_f, batch_dyn_g, Lg_h, Lf_h = jax_vmap(ft.partial(self.get_single_h_gradient, cbf_params=params))(b_graph)
        return batch_h, batch_h_x, batch_dyn_f, batch_dyn_g, Lg_h, Lf_h
    
    @ft.partial(jax.jit, static_argnums=(0,))
    def get_batch_V_gradient(self, b_graph: GraphsTuple, params) : #-> Action
        batch_V, batch_V_x = jax_vmap(ft.partial(self.get_single_V_gradient, value_params=params))(b_graph)
        return batch_V, batch_V_x
    
    def update_nets(self, rollout: Rollout, rollout_random: Rollout, safe_mask, unsafe_mask, num_step: int):
        update_info = {}

        n_chunks = 8 
        batch_size = len(rollout.graph.states)
        chunk_size = batch_size // n_chunks 

        b_u_kkt = []
        for ii in range(n_chunks):
            graph = jtu.tree_map(lambda x: x[ii * chunk_size: (ii + 1) * chunk_size], rollout.graph)
            c, V_x, lag_term, lagrange = self.get_b_u_kkt(graph, self.cbf_tgt.params, self.actor_train_state.params, self.value_train_state.params, num_step=num_step)
            b_u_kkt.append(jax2np(c))
        b_u_kkt = tree_merge(b_u_kkt)
        
        batch_orig = Batch(rollout.graph, safe_mask, unsafe_mask, b_u_kkt) # batch for cbf
        batch_value = Batch(rollout_random.graph, None, None, None) # batch for value
        
        length_rollout = rollout.length if rollout.length < rollout_random.length else rollout_random.length
        for i_epoch in range(self.inner_epoch):
            # shuffle data
            idx = self.rng.choice(rollout.length, size=length_rollout, replace=False)
            idx_r = self.rng.choice(rollout_random.length, size=length_rollout, replace=False)
            
            batch_idx = np.stack(np.array_split(idx, idx.shape[0] // self.batch_size), axis=0)
            batch_idx_r = np.stack(np.array_split(idx_r, idx_r.shape[0] // self.batch_size), axis=0)
            
            batch = jtu.tree_map(lambda x: x[batch_idx], batch_orig)
            batch_r = jtu.tree_map(lambda x: x[batch_idx_r], batch_value)
            
            cbf_train_state, actor_train_state, value_train_state, update_info = self.update_inner(
                self.cbf_train_state, self.actor_train_state, self.value_train_state, batch, batch_r, num_step, self.training_state
                )
            
            # log
            update_info['cbf_lr'] = self.cbf_lr_scheduler(cbf_train_state.step)
            update_info['actor_lr'] = self.actor_lr_scheduler(actor_train_state.step)
            update_info['value_lr'] = self.value_lr_scheduler(value_train_state.step)
            update_info['warm_up'] = 0
            update_info['update_actor'] = 0
            update_info['update_value'] = 0
            update_info['save_critic'] = self.save_critc

            # update cbf
            self.cbf_train_state = cbf_train_state
            
            # update value and actor
            if self.training_state == TrainingState.WARMUP:
                update_info['warm_up'] = 1
                self.actor_train_state = actor_train_state
                self.value_train_state = value_train_state
                self.training_count += 1
                if self.training_count >= 1000 or update_info['loss/h'] < 0:
                    self.save_actor_state = actor_train_state
                    self.save_value_state = value_train_state
                    self.training_state = TrainingState.UPDATEVALUE
                    self.training_count = 0
                    self.save_critc = update_info['loss/loss_critic']

            elif self.training_state == TrainingState.UPDATEACTIOR:
                update_info['update_actor'] = 1
                self.actor_train_state = actor_train_state
                self.training_count += 1
                # update successful
                if update_info['loss/h'] < 0 : 
                    self.save_actor_state = actor_train_state
                    self.training_state = TrainingState.UPDATEVALUE
                    self.training_count = 0
                # update failed
                elif self.training_count >= self.training_max: 
                    self.actor_train_state = self.save_actor_state.replace(step=self.actor_train_state.step)
                    self.training_state = TrainingState.UPDATEVALUE
                    self.training_count = 0
                    
            elif self.training_state == TrainingState.UPDATEVALUE:
                update_info['update_value'] = 1
                self.value_train_state = value_train_state
                self.training_count += 1
                # update successful
                if update_info['loss/loss_critic'] < self.save_critc*1.1: 
                    self.save_critc = update_info['loss/loss_critic']
                    self.save_value_state = value_train_state
                    self.training_state = TrainingState.UPDATEACTIOR
                    self.training_count = 0
                # update failed
                elif self.training_count >= self.training_max:
                    self.value_train_state = self.save_value_state.replace(step=self.value_train_state.step)
                    self.training_state = TrainingState.UPDATEACTIOR
                    self.training_count = 0
                    
        self.cbf_tgt = self.update_tgt(self.cbf_tgt, self.cbf_train_state, 0.5)
        
        return update_info

    def sample_batch(self, rollout: Rollout, safe_mask, unsafe_mask):
        if self.buffer.length > self.batch_size:
            # sample from memory
            memory, safe_mask_memory, unsafe_mask_memory = self.buffer.sample(rollout.length)
            unsafe_memory, safe_mask_unsafe_memory, unsafe_mask_unsafe_memory = self.unsafe_buffer.sample(
                rollout.length * rollout.time_horizon)

            # append new data to memory
            self.buffer.append(rollout, safe_mask, unsafe_mask)
            unsafe_multi_mask = unsafe_mask.max(axis=-1)
            self.unsafe_buffer.append(
                jtu.tree_map(lambda x: x[unsafe_multi_mask], rollout),
                safe_mask[unsafe_multi_mask],
                unsafe_mask[unsafe_multi_mask]
            )

            # get update data
            # (b, T)
            rollout = tree_merge([memory, rollout])
            safe_mask = tree_merge([safe_mask_memory, safe_mask])
            unsafe_mask = tree_merge([unsafe_mask_memory, unsafe_mask])

            # (b, T) -> (b * T, )
            rollout = jtu.tree_map(lambda x: merge01(x), rollout)
            safe_mask = merge01(safe_mask)
            unsafe_mask = merge01(unsafe_mask)
            rollout_batch = tree_merge([unsafe_memory, rollout])
            safe_mask_batch = tree_merge([safe_mask_unsafe_memory, safe_mask])
            unsafe_mask_batch = tree_merge([unsafe_mask_unsafe_memory, unsafe_mask])
        else:
            self.buffer.append(rollout, safe_mask, unsafe_mask)
            unsafe_multi_mask = unsafe_mask.max(axis=-1)
            self.unsafe_buffer.append(
                jtu.tree_map(lambda x: x[unsafe_multi_mask], rollout),
                safe_mask[unsafe_multi_mask],
                unsafe_mask[unsafe_multi_mask]
            )

            # (b, T) -> (b * T, )
            rollout_batch = jtu.tree_map(lambda x: merge01(x), rollout)
            safe_mask_batch = merge01(safe_mask)
            unsafe_mask_batch = merge01(unsafe_mask)

        return rollout_batch, safe_mask_batch, unsafe_mask_batch
    
    def update(self, rollout: Rollout, rollout_random: Rollout, num_step: int=0) -> dict:
        key, self.key = jr.split(self.key)

        # (n_collect, T)
        unsafe_mask = jax_vmap(jax_vmap(self._env.unsafe_mask))(rollout.graph)
        safe_mask = self.safe_mask(unsafe_mask)
        safe_mask, unsafe_mask = jax2np(safe_mask), jax2np(unsafe_mask)
        
        rollout_np = jax2np(rollout)
        rollout_random_np = jax2np(rollout_random)


        del rollout, rollout_random
        rollout_batch, safe_mask_batch, unsafe_mask_batch = self.sample_batch(rollout_np, safe_mask, unsafe_mask) 
        rollout_random_batch = jtu.tree_map(lambda x: merge01(x), rollout_random_np)

        # inner loop
        update_info = self.update_nets(rollout_batch, rollout_random_batch, safe_mask_batch, unsafe_mask_batch, num_step=num_step)

        return update_info

    def get_kkt_action(
            self,
            graph: GraphsTuple,
            num_step: int,
            cbf_params=None,
            value_params=None,
            actor_params=None,
    ) -> [Action, Array]:
        assert graph.is_single
        lagrange, V_x, V, h_x, h_dot_m, h, dyn_f, dyn_g = self.get_single_lagrange(graph, cbf_params, actor_params, value_params)
        
        R_inv = jnp.linalg.inv(self._env._R)
        lagrange = jnp.clip(lagrange,0,10)
        lag_term = lagrange[:, None, None] * h_x 
        lag_term = jnp.clip(lag_term,-0.5,0.5)
        g_hl = np.zeros((12, 4))  
        g_hl[:4, :] = np.eye(4)  
        dyn_g_hl = np.tile(g_hl, (self.n_agents, 1, 1)) 
        R_inv_g_T_hl = ei.einsum(R_inv, dyn_g_hl, "nx ne, agent_j nxx ne -> agent_j nx nxx")
        R_inv_g_T = ei.einsum(R_inv, dyn_g, "nx ne, agent_j nxx ne -> agent_j nx nxx")
        u_kk_v = ei.einsum(-0.5*R_inv_g_T_hl, V_x, "agent_i nx nxx, agent_j agent_i nxx -> agent_j nx")
        u_kk_s = ei.einsum(-0.5*R_inv_g_T, -lag_term, "agent_i nx nxx, agent_j agent_i nxx -> agent_j nx")
        u_kkt = u_kk_v + u_kk_s
        u_kkt_clip =self._env.clip_action(u_kkt)
        
        return u_kkt_clip, V_x, lag_term, lagrange

    def get_single_lagrange(
                self,
                graph: GraphsTuple,
                cbf_params=None,
                actor_params=None,
                value_params=None
        ):
            assert graph.is_single
            V, V_x = self.get_single_V_gradient(graph, value_params)
            V_xi = jnp.diagonal(V_x, axis1=0, axis2=1).T
            h, h_x, dyn_f, dyn_g, Lg_h, Lf_h = self.get_single_h_gradient(graph, cbf_params)
            h_xi = jnp.diagonal(h_x, axis1=0, axis2=1).T
            R_ii = jnp.array([self._env._R for _ in range(self.n_agents)])
            R_inv = jnp.linalg.inv(R_ii)
            
            # get H_i
            g_R_inv = ei.einsum(dyn_g, R_inv, "agent nx nu, agent nu ne -> agent nx ne") 
            g_R_inv_gT = ei.einsum(g_R_inv, dyn_g, "agent nx ne, agent nxx ne -> agent nx nxx") 
            h_xT_g_R_inv_gT = ei.einsum(h_xi, g_R_inv_gT,"agent nx, agent nx nxx-> agent nxx") 
            H_i = 0.5*ei.einsum(h_xT_g_R_inv_gT, h_xi, "agent nx, agent nx-> agent") 
            
            # get action
            action_fn = ft.partial(self.act, params=actor_params)
            action = action_fn(graph)
            
            # get C_i
            h_xT_f= ei.einsum(h_xi, dyn_f, "agent nx, agent nx -> agent") 
            g_hl = np.zeros((12, 4))  
            g_hl[:4, :] = np.eye(4)  
            dyn_g_hl = np.tile(g_hl, (self.n_agents, 1, 1)) 
            g_R_inv_gT_hl = ei.einsum(g_R_inv, dyn_g_hl, "agent nx ne, agent nxx ne -> agent nx nxx") 
            h_xT_g_R_inv_gT_hl = ei.einsum(h_xi, g_R_inv_gT_hl,"agent nx, agent nx nxx-> agent nxx") 
            h_xT_g_R_inv_gT_V_x = ei.einsum(h_xT_g_R_inv_gT_hl, V_xi, "agent nx, agent nx-> agent") 
            agent_term = h_xT_f - 0.5*h_xT_g_R_inv_gT_V_x 
            
            forward_fn = self._env.forward_graph
            next_graph = forward_fn(graph, action)
            cbf_fn = ft.partial(self.cbf.get_cbf, cbf_params)
            h_next = cbf_fn(next_graph)[:,0]
            h_dot = (h_next - h) / self._env.dt
            Lg_h_u = ei.einsum(Lg_h, action, "agent_i agent_j nu, agent_i nu -> agent_i agent_j")
            Lg_h_u = jnp.diagonal(Lg_h_u, axis1=0, axis2=1)
            C_i = h_dot - Lg_h_u - 0.5*h_xT_g_R_inv_gT_V_x + self.alpha*h
            condition = C_i < 0.8
            lagrange = jnp.where(condition, -C_i / (H_i+1e-8), 0) 

            return lagrange, V_x, V, h_x, h_x, h, dyn_f, dyn_g

    def get_single_h_gradient(
                self,
                graph: GraphsTuple,
                cbf_params=None,
        ) : #-> [Action, Array]
            assert graph.is_single
            agent_node_mask = graph.node_type == 0
            agent_node_id = mask2index(agent_node_mask, self.n_agents)
            
            def h_aug(new_agent_state: State) -> Array:
                new_state = graph.states.at[agent_node_id].set(new_agent_state)
                new_graph = self._env.add_edge_feats(graph, new_state)
                return self.get_cbf(new_graph, params=cbf_params)

            agent_state = graph.type_states(type_idx=0, n_type=self.n_agents)
            h = h_aug(agent_state).squeeze(-1)
            h_x = jax.jacobian(h_aug)(agent_state).squeeze(1)
            dyn_f, dyn_g = self._env.control_affine_dyn(agent_state)
            Lf_h = ei.einsum(h_x, dyn_f, "agent_i agent_j nx, agent_j nx -> agent_i agent_j")
            Lg_h = ei.einsum(h_x, dyn_g, "agent_i agent_j nx, agent_j nx nu -> agent_i agent_j nu")
            
            return h, h_x, dyn_f, dyn_g, Lg_h, Lf_h
    
    def get_single_V_gradient(
                self,
                graph: GraphsTuple,
                value_params=None,
        ) : #-> [Action, Array]
            assert graph.is_single
            agent_node_mask = graph.node_type == 0
            agent_node_id = mask2index(agent_node_mask, self.n_agents)
            def V_aug(new_agent_state: State) -> Array:
                new_state = graph.states.at[agent_node_id].set(new_agent_state)
                new_graph = self._env.add_edge_feats(graph, new_state)
                return self.val(new_graph, params=value_params)
            
            agent_state = graph.type_states(type_idx=0, n_type=self.n_agents)
            V = V_aug(agent_state).squeeze(-1)
            V_x =jax.jacobian(V_aug)(agent_state).squeeze(1)
            return V, V_x
    

    @ft.partial(jax.jit, static_argnums=(0,-1), donate_argnums=(1, 2))
    def update_inner(
            self, cbf_train_state: TrainState, actor_train_state: TrainState, value_train_state: TrainState, batch: Batch, batch_value: Batch, num_step: int, t_state: TrainingState
    ) -> tuple[TrainState, TrainState, dict]:
        
        def update_fn(carry, b: Tuple[Batch, Batch], training_state: TrainingState):
            minibatch, minibatch_value = b
            cbf, actor, value = carry
            safe_mask_batch = merge01(minibatch.safe_mask)
            unsafe_mask_batch = merge01(minibatch.unsafe_mask)

            def get_loss(cbf_params: Params, actor_params: Params, value_params: Params) -> Tuple[Array, dict]:
                #get CBF
                cbf_fn = jax_vmap(ft.partial(self.cbf.get_cbf, cbf_params))
                cbf_fn_no_grad = jax_vmap(ft.partial(self.cbf.get_cbf, jax.lax.stop_gradient(cbf_params)))
                h_m = cbf_fn(minibatch.graph).squeeze()
                h = merge01(h_m)

                unsafe_data_ratio = jnp.mean(unsafe_mask_batch)
                h_unsafe = jnp.where(unsafe_mask_batch, h, -jnp.ones_like(h) * self.eps * 2)
                max_val_unsafe = jax.nn.relu(h_unsafe + self.eps)
                loss_unsafe = jnp.sum(max_val_unsafe) / (jnp.count_nonzero(unsafe_mask_batch) + 1e-6)
                acc_unsafe_mask = jnp.where(unsafe_mask_batch, h, jnp.ones_like(h))
                acc_unsafe = (jnp.sum(jnp.less(acc_unsafe_mask, 0)) + 1e-6) / (jnp.count_nonzero(unsafe_mask_batch) + 1e-6)

                # safe region h(x) > 0
                h_safe = jnp.where(safe_mask_batch, h, jnp.ones_like(h) * self.eps * 2)
                max_val_safe = jax.nn.relu(-h_safe + self.eps)
                loss_safe = jnp.sum(max_val_safe) / (jnp.count_nonzero(safe_mask_batch) + 1e-6)
                acc_safe_mask = jnp.where(safe_mask_batch, h, -jnp.ones_like(h))
                acc_safe = (jnp.sum(jnp.greater(acc_safe_mask, 0)) + 1e-6) / (jnp.count_nonzero(safe_mask_batch) + 1e-6)

                # get neural network actions
                action_fn = jax.vmap(ft.partial(self.act, params=actor_params))
                action = action_fn(minibatch.graph)
                
                # get next graph
                forward_fn = jax_vmap(self._env.forward_graph)
                next_graph = forward_fn(minibatch.graph, action)
                h_next_m = cbf_fn(next_graph).squeeze()
                h_next = merge01(h_next_m)
                h_dot = (h_next - h) / self._env.dt

                # stop gradient and get next graph
                h_no_grad = jax.lax.stop_gradient(h)
                h_next_no_grad = merge01(cbf_fn_no_grad(next_graph).squeeze())
                h_dot_no_grad = (h_next_no_grad - h_no_grad) / self._env.dt

                # h_dot + alpha * h > 0 (backpropagate to action, and backpropagate to h when labeled)
                labeled_mask = jnp.logical_or(unsafe_mask_batch, safe_mask_batch)
                max_val_h_dot = jax.nn.relu(-h_dot - self.alpha * h + self.eps)
                max_val_h_dot_no_grad = jax.nn.relu(-h_dot_no_grad - self.alpha * h + self.eps)
                max_val_h_dot = jnp.where(labeled_mask, max_val_h_dot, max_val_h_dot_no_grad)
                loss_h_dot = jnp.mean(max_val_h_dot)
                acc_h_dot = jnp.mean(jnp.greater(h_dot + self.alpha * h, 0))

                
                # value loss
                action_fn = jax.vmap(ft.partial(self.act, params=actor_params))
                action_v = action_fn(minibatch_value.graph)
                device_graphs = jax.tree_util.tree_map(lambda x: x, minibatch_value.graph)
                compute_states = jax.vmap(lambda graph: graph.type_states(type_idx=0, n_type=self.n_agents))
                agent = compute_states(device_graphs)
                goal = minibatch_value.graph.env_states.goal
                
                error = agent-goal 
                norm_error = jnp.linalg.norm(error, axis=-1, keepdims=True)
                error_max = jnp.abs(error / (norm_error+1e-8) * self._env._params["comm_radius"])
                error = jnp.clip(error, -error_max, error_max) 
                error = error[...,0:4]
                
                xdot_fn = jax.vmap(ft.partial(self._env.agent_xdot_hl))
                xdot=xdot_fn(agent, action_v)
                
                b_V, b_V_x = self.get_batch_V_gradient(minibatch_value.graph, value_params)
                
                V_N_i_sum = ei.einsum(b_V_x[...,0:4], xdot, "n_graghs agent_i agent_j nx, n_graghs agent_j nx -> n_graghs agent_i") 
                e_T_Q_e = ei.einsum(error, self._env._Q_hl, error, "n_graghs agent_i nx, nx ne, n_graghs agent_i ne-> n_graghs agent_i")  
                u_R_u =ei.einsum(action_v, self._env._R_hl, action_v, "n_graghs agent_i nx, nx na, n_graghs agent_i na-> n_graghs agent_i")  
                Bell_error=(e_T_Q_e + u_R_u) + V_N_i_sum

                loss_critic = jnp.mean(jnp.square(Bell_error).sum(axis=-1))
                loss_actor = jnp.mean((Bell_error).sum(axis=-1))
                loss_h = jnp.max(V_N_i_sum.sum(axis=-1))
                
                loss_low = b_V - jnp.linalg.norm(error, axis=-1)*50
                loss_low = jnp.where(loss_low>0, 0, loss_low)
                loss_high = jnp.linalg.norm(error, axis=-1)*100 - b_V
                loss_high = jnp.where(loss_high>0, 0, loss_high)
                loss_equ = jnp.mean(jnp.square(loss_low).sum(axis=-1)) + jnp.mean(jnp.square(loss_high).sum(axis=-1))
                
                loss_value = loss_critic + loss_equ*0.1
                
                # actor loss
                assert action.shape == minibatch.u_kkt.shape
                loss_action = jnp.mean(jnp.square(action - minibatch.u_kkt).sum(axis=-1))

                # total loss
                total_loss = (
                    self.loss_value_coef * loss_value
                       + self.loss_action_coef * loss_action
                       + self.loss_unsafe_coef * loss_unsafe
                       + self.loss_safe_coef * loss_safe
                       + self.loss_h_dot_coef * loss_h_dot
                )
                return total_loss, {'loss/total': total_loss,
                                    'loss/coef_value': self.loss_value_coef * loss_value,
                                    'loss/coef_action': self.loss_action_coef * loss_action,
                                    'loss/coef_h': self.loss_unsafe_coef * loss_unsafe + self.loss_safe_coef * loss_safe + self.loss_h_dot_coef * loss_h_dot,
                                    'loss/value': loss_value,
                                    'loss/loss_critic':loss_critic,
                                    'loss/loss_equ': loss_equ,
                                    'loss/loss_actor': loss_actor,
                                    'loss/action': loss_action,
                                    'loss/unsafe': loss_unsafe,
                                    'loss/safe': loss_safe,
                                    'loss/h_dot': loss_h_dot,
                                    'loss/h': loss_h,
                                    'acc/unsafe': acc_unsafe,
                                    'acc/safe': acc_safe,
                                    'acc/h_dot': acc_h_dot,
                                    'acc/unsafe_data_ratio': unsafe_data_ratio,
                                    'stepp': num_step,
                                    }
            
            (loss, loss_info), (grad_cbf, grad_actor, grad_value) = jax.value_and_grad(
                get_loss, has_aux=True, argnums=(0, 1, 2))(cbf.params, actor.params, value.params)
            
            # clip gradient
            grad_cbf, grad_cbf_norm = compute_norm_and_clip(grad_cbf, self.max_grad_norm)
            grad_actor, grad_actor_norm = compute_norm_and_clip(grad_actor, self.max_grad_norm)
            grad_value, grad_value_norm = compute_norm_and_clip(grad_value, self.max_grad_norm)
            # update
            cbf = cbf.apply_gradients(grads=grad_cbf)
            actor = actor.apply_gradients(grads=grad_actor)
            value = value.apply_gradients(grads=grad_value)
            grad_info = {'grad_norm/cbf': grad_cbf_norm, 'grad_norm/actor': grad_actor_norm, 'grad_norm/value': grad_value_norm,'grad/value': grad_value,}
            return (cbf, actor, value), grad_info | loss_info
        
        train_state = (cbf_train_state, actor_train_state, value_train_state)
        update_fn2 = ft.partial(update_fn, training_state = t_state)
        (cbf_train_state, actor_train_state, value_train_state), info = lax.scan(update_fn2, train_state, (batch, batch_value))

        info = jtu.tree_map(lambda x: x[-1], info)
        return cbf_train_state, actor_train_state, value_train_state, info

    def save(self, save_dir: str, step: int):
        model_dir = os.path.join(save_dir, str(step))
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        pickle.dump(self.actor_train_state.params, open(os.path.join(model_dir, 'actor.pkl'), 'wb'))
        pickle.dump(self.value_train_state.params, open(os.path.join(model_dir, 'value.pkl'), 'wb'))
        pickle.dump(self.cbf_train_state.params, open(os.path.join(model_dir, 'cbf.pkl'), 'wb'))

    def load(self, load_dir: str, step: int):
        path = os.path.join(load_dir, str(step))

        self.actor_train_state = \
            self.actor_train_state.replace(params=pickle.load(open(os.path.join(path, 'actor.pkl'), 'rb')))
        self.value_train_state = \
            self.value_train_state.replace(params=pickle.load(open(os.path.join(path, 'value.pkl'), 'rb')))
        self.cbf_train_state = \
            self.cbf_train_state.replace(params=pickle.load(open(os.path.join(path, 'cbf.pkl'), 'rb')))
