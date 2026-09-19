#!/usr/bin/env python3
"""Bidirectional Neural Network ↔ LLM/SLM Communication (SAR-only).

Provides bidirectional communication between neural networks and LLMs/SLMs:
- Neural Network → LLM/SLM: Neural outputs → structured LLM prompts
- LLM/SLM → Neural Network: LLM outputs → structured neural network inputs

SAR-only: All communication constrained to SAR primitives (navigate_to, hover,
drop_payload, return_home). No weaponization, targeting, or kinetic effectors.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from enum import Enum
from abc import ABC, abstractmethod
import threading
import queue
import time

import numpy as np
import torch
import torch.nn as nn

# Import existing SAR components
from src.control.language_interface import (
    parse_language, language_to_command, text_to_task_embedding,
    parse_with_llm, TextEncoder, ProjectionHead, TASK_EMBED_DIM,
    TASK_NAMES, NUM_TASKS
)
from src.control.language_to_task import (
    text_to_task, intent_to_task, TASK_ID_TO_SKILL
)
from src.control.safety_filter import allow
from src.rl.policies import ActorCritic, OBS_DIM, ACT_DIM, DEVICE
from src.rl.local_policy import LocalPolicy, HierarchicalPolicy
from src.rl.ppo_nav import HierarchicalPolicy as PPOHierarchicalPolicy


# ============================================================================
# Message Types & Protocols
# ============================================================================

class MessageType(Enum):
    """Message types for NN ↔ LLM communication."""
    # NN → LLM
    NN_OBSERVATION = "nn_observation"          # NN sends observation
    NN_ACTION = "nn_action"                    # NN sends action taken
    NN_STATE = "nn_state"                      # NN sends internal state
    NN_REWARD = "nn_reward"                    # NN sends reward signal
    NN_EPISODE_END = "nn_episode_end"          # NN signals episode end
    NN_HELP_REQUEST = "nn_help_request"        # NN asks for help
    
    # LLM → NN
    LLM_COMMAND = "llm_command"                # LLM sends command
    LLM_PLAN = "llm_plan"                      # LLM sends plan
    LLM_CORRECTION = "llm_correction"          # LLM corrects NN
    LLM_GOAL = "llm_goal"                      # LLM sets goal
    LLM_CORRECTION_FEEDBACK = "llm_correction_feedback"  # LLM corrects NN action
    LLM_QUERY = "llm_query"                    # LLM queries NN state
    
    # Bidirectional
    HEARTBEAT = "heartbeat"
    ACK = "ack"
    ERROR = "error"


@dataclass
class NNMessage:
    """Message from Neural Network to LLM."""
    msg_type: MessageType
    timestamp: float = field(default_factory=time.time)
    msg_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    session_id: str = ""
    episode: int = 0
    step: int = 0
    
    # Neural network state
    observation: Optional[np.ndarray] = None
    action: Optional[np.ndarray] = None
    reward: float = 0.0
    done: bool = False
    info: Dict = field(default_factory=dict)
    
    # Neural network internal state
    hidden_state: Optional[np.ndarray] = None
    value_estimate: Optional[float] = None
    action_logprob: Optional[float] = None
    entropy: Optional[float] = None
    
    # Episode info
    episode_reward: float = 0.0
    episode_length: int = 0
    episode_done: bool = False
    
    # Help request
    help_reason: str = ""
    uncertainty: float = 0.0
    
    def to_dict(self) -> Dict:
        d = asdict(self)
        # Convert numpy arrays to lists for JSON serialization
        for k, v in d.items():
            if isinstance(v, np.ndarray):
                d[k] = v.tolist()
            elif isinstance(v, np.generic):
                d[k] = v.item()
        return d
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'NNMessage':
        # Convert lists back to numpy arrays
        for k, v in d.items():
            if isinstance(v, list) and k in ['observation', 'action', 'hidden_state']:
                d[k] = np.array(v, dtype=np.float32)
        return cls(**d)


@dataclass
class LLMMessage:
    """Message from LLM to Neural Network."""
    msg_type: MessageType
    timestamp: float = field(default_factory=time.time)
    msg_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    session_id: str = ""
    request_id: str = ""
    
    # LLM command/plan
    command_type: str = ""  # navigate_to, hover, drop_payload, return_home, etc.
    params: Dict = field(default_factory=dict)
    constraints: Dict = field(default_factory=dict)
    
    # Plan
    plan: List[Dict] = field(default_factory=list)  # List of waypoints/goals
    plan_id: str = ""
    
    # Correction
    correction_type: str = ""  # action_correction, goal_correction, plan_correction
    correction_data: Dict = field(default_factory=dict)
    
    # Goal
    goal_type: str = ""  # navigate_to, hover, drop_payload, return_home
    goal_params: Dict = field(default_factory=dict)
    goal_constraints: Dict = field(default_factory=dict)
    
    # Query
    query_type: str = ""  # state_query, uncertainty_query, help_request
    query_params: Dict = field(default_factory=dict)
    
    # Metadata
    confidence: float = 1.0
    priority: int = 1
    ttl: float = 30.0  # Time to live in seconds
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'LLMMessage':
        return cls(**d)
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict())


@dataclass
class CommunicationConfig:
    """Configuration for NN-LLM communication."""
    # Timeouts
    llm_timeout_s: float = 10.0
    nn_timeout_s: float = 5.0
    heartbeat_interval_s: float = 1.0
    
    # Queue sizes
    nn_to_llm_queue_size: int = 100
    llm_to_nn_queue_size: int = 100
    
    # Retry
    max_retries: int = 3
    retry_delay_s: float = 0.5
    
    # Validation
    validate_sar_only: bool = True
    require_ack: bool = True
    
    # Serialization
    use_binary: bool = False  # If True, use msgpack; else JSON


# ============================================================================
# Communication Channels
# ============================================================================

class CommunicationChannel(ABC):
    """Abstract base class for communication channels."""
    
    @abstractmethod
    def send(self, message: Union[NNMessage, LLMMessage]) -> bool:
        """Send a message. Returns True if sent successfully."""
        pass
    
    @abstractmethod
    def receive(self, timeout: float = 1.0) -> Optional[Union[NNMessage, LLMMessage]]:
        """Receive a message with timeout. Returns None if timeout."""
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Close the channel."""
        pass


class InProcessChannel(CommunicationChannel):
    """In-process communication channel using thread-safe queues."""
    
    def __init__(self, config: CommunicationConfig):
        self.config = config
        self.nn_to_llm_queue = queue.Queue(maxsize=config.nn_to_llm_queue_size)
        self.llm_to_nn_queue = queue.Queue(maxsize=config.llm_to_nn_queue_size)
        self._closed = False
        self._lock = threading.Lock()
    
    def send(self, message: Union[NNMessage, LLMMessage]) -> bool:
        if self._closed:
            return False
        try:
            if isinstance(message, NNMessage):
                self.nn_to_llm_queue.put(message, timeout=1.0)
            elif isinstance(message, LLMMessage):
                self.llm_to_nn_queue.put(message, timeout=1.0)
            else:
                return False
            return True
        except queue.Full:
            return False
    
    def receive(self, timeout: float = 1.0, from_nn: bool = True) -> Optional[Union[NNMessage, LLMMessage]]:
        if self._closed:
            return None
        try:
            if from_nn:
                return self.nn_to_llm_queue.get(timeout=timeout)
            else:
                return self.llm_to_nn_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def close(self) -> None:
        with self._lock:
            self._closed = True


class ZMQChannel(CommunicationChannel):
    """ZeroMQ-based channel for inter-process communication."""
    
    def __init__(self, config: CommunicationConfig, 
                 nn_to_llm_addr: str = "tcp://127.0.0.1:5555",
                 llm_to_nn_addr: str = "tcp://127.0.0.1:5556"):
        self.config = config
        self.context = None
        self.nn_to_llm_socket = None
        self.llm_to_nn_socket = None
        self.nn_to_llm_addr = nn_to_llm_addr
        self.llm_to_nn_addr = llm_to_nn_addr
        self._init_sockets()
    
    def _init_sockets(self):
        import zmq
        self.context = zmq.Context()
        
        # NN → LLM (PUSH/PULL)
        self.nn_to_llm_socket = self.context.socket(zmq.PUSH)
        self.nn_to_llm_socket.bind(self.nn_to_llm_addr)
        
        # LLM → NN (PUB/SUB)
        self.llm_to_nn_socket = self.context.socket(zmq.PUB)
        self.llm_to_nn_socket.bind(self.llm_to_nn_addr)
    
    def send(self, message: Union[NNMessage, LLMMessage]) -> bool:
        try:
            data = message.to_json().encode('utf-8')
            if isinstance(message, NNMessage):
                self.nn_to_llm_socket.send(data, flags=zmq.NOBLOCK)
            else:
                self.llm_to_nn_socket.send(data, flags=zmq.NOBLOCK)
            return True
        except Exception:
            return False
    
    def receive(self, timeout: float = 1.0, from_nn: bool = True) -> Optional[Union[NNMessage, LLMMessage]]:
        import zmq
        try:
            if from_nn:
                # For NN messages, we'd need a PULL socket (not implemented here)
                return None
            else:
                # LLM → NN via SUB
                self.llm_to_nn_socket.setsockopt(zmq.RCVTIMEO, int(1000))
                data = self.llm_to_nn_socket.recv(flags=zmq.NOBLOCK)
                msg_dict = json.loads(data.decode('utf-8'))
                return LLMMessage.from_dict(msg_dict)
        except Exception:
            return None
    
    def close(self) -> None:
        if self.nn_to_llm_socket:
            self.nn_to_llm_socket.close()
        if self.llm_to_nn_socket:
            self.llm_to_nn_socket.close()
        if self.context:
            self.context.term()


# ============================================================================
# Neural Network Interface
# ============================================================================

class NeuralNetworkInterface:
    """Interface for neural network to communicate with LLM."""
    
    def __init__(self, config: CommunicationConfig, channel: CommunicationChannel,
                 policy: Union[ActorCritic, LocalPolicy, HierarchicalPolicy],
                 session_id: str = ""):
        self.config = config
        self.channel = channel
        self.policy = policy
        self.session_id = str(uuid.uuid4())[:8] if not session_id else session_id
        self.episode = 0
        self.step_count = 0
        self.episode_reward = 0.0
        self.episode_length = 0
        self.last_observation = None
        self.last_action = None
        self.last_value = None
        self.last_logprob = None
        self.episode_reward = 0.0
        self.episode_length = 0
        self.pending_acks: Dict[str, float] = {}  # msg_id -> timestamp
        self.pending_requests: Dict[str, Tuple[float, Callable]] = {}  # request_id -> (timestamp, callback)
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
    
    def start(self):
        """Start the communication loop."""
        self._running = True
        self._thread = threading.Thread(target=self._communication_loop, daemon=True)
        self._thread.start()
    
    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
    
    def _communication_loop(self):
        """Main communication loop."""
        while self._running:
            # Send heartbeat
            self._send_heartbeat()
            
            # Check for incoming LLM messages
            msg = self.channel.receive(timeout=0.1, from_nn=False)
            if msg:
                self._handle_llm_message(msg)
            
            # Check for pending ACKs
            self._check_pending_acks()
            
            # Check pending requests
            self._check_pending_requests()
            
            time.sleep(0.01)  # 100Hz loop
    
    def _send_heartbeat(self):
        msg = NNMessage(
            msg_type=MessageType.HEARTBEAT,
            session_id=self.session_id,
            episode=self.episode,
            step=self.step_count,
        )
        self.channel.send(msg)
    
    def send_observation(self, observation: np.ndarray, 
                         action: Optional[np.ndarray] = None,
                         reward: float = 0.0,
                         done: bool = False,
                         info: Dict = None,
                         hidden_state: Optional[np.ndarray] = None,
                         value: Optional[float] = None,
                         logprob: Optional[float] = None,
                         entropy: Optional[float] = None):
        """Send observation to LLM."""
        self.last_observation = observation
        if action is not None:
            self.last_action = action
        
        msg = NNMessage(
            msg_type=MessageType.NN_OBSERVATION,
            session_id=self.session_id,
            episode=self.episode,
            step=self.step_count,
            observation=observation,
            action=action,
            reward=reward,
            done=done,
            info=info or {},
            hidden_state=hidden_state,
            value_estimate=value,
            action_logprob=logprob,
            entropy=entropy,
            episode_reward=self.episode_reward,
            episode_length=self.episode_length,
        )
        self.channel.send(msg)
        self.step_count += 1
        self.episode_reward += reward
        self.episode_length += 1
        
        if done:
            self._end_episode()
    
    def _end_episode(self):
        msg = NNMessage(
            msg_type=MessageType.NN_EPISODE_END,
            session_id=self.session_id,
            episode=self.episode,
            step=self.step_count,
            episode_reward=self.episode_reward,
            episode_length=self.episode_length,
        )
        self.channel.send(msg)
        self.episode += 1
        self.step_count = 0
        self.episode_reward = 0.0
        self.episode_length = 0
    
    def request_help(self, reason: str, uncertainty: float = 1.0):
        """Request help from LLM."""
        msg = NNMessage(
            msg_type=MessageType.NN_HELP_REQUEST,
            session_id=self.session_id,
            episode=self.episode,
            step=self.step_count,
            help_reason=reason,
            uncertainty=uncertainty,
        )
        self.channel.send(msg)
    
    def _handle_llm_message(self, msg: LLMMessage):
        """Handle incoming LLM message."""
        if msg.msg_type == MessageType.LLM_COMMAND:
            self._execute_command(msg)
        elif msg.msg_type == MessageType.LLM_PLAN:
            self._execute_plan(msg)
        elif msg.msg_type == MessageType.LLM_CORRECTION:
            self._apply_correction(msg)
        elif msg.msg_type == MessageType.LLM_GOAL:
            self._set_goal(msg)
        elif msg.msg_type == MessageType.LLM_QUERY:
            self._answer_query(msg)
        elif msg.msg_type == MessageType.ACK:
            self._handle_ack(msg)
        elif msg.msg_type == MessageType.ERROR:
            print(f"LLM Error: {msg.params.get('error', 'Unknown error')}")
    
    def _execute_command(self, msg: LLMMessage):
        """Execute LLM command."""
        # Convert LLM command to policy action
        if msg.command_type == "navigate_to":
            # Set goal for policy
            pass
        elif msg.command_type == "hover":
            pass
        elif msg.command_type == "drop_payload":
            pass
        elif msg.command_type == "return_home":
            pass
        
        # Send ACK
        self._send_ack(msg.msg_id)
    
    def _execute_plan(self, msg: LLMMessage):
        """Execute multi-step plan."""
        # Store plan for execution
        pass
    
    def _apply_correction(self, msg: LLMMessage):
        """Apply correction from LLM."""
        pass
    
    def _set_goal(self, msg: LLMMessage):
        """Set new goal from LLM."""
        pass
    
    def _answer_query(self, msg: LLMMessage):
        """Answer LLM query about NN state."""
        response = LLMMessage(
            msg_type=MessageType.LLM_QUERY,
            session_id=self.session_id,
            request_id=msg.request_id,
            query_type=msg.query_type,
            query_params={
                "state": self.last_observation.tolist() if self.last_observation is not None else None,
                "episode": self.episode,
                "step": self.step_count,
                "episode_reward": self.episode_reward,
            }
        )
        self.channel.send(msg)
    
    def _send_ack(self, msg_id: str):
        ack = LLMMessage(
            msg_type=MessageType.ACK,
            session_id=self.session_id,
            request_id=msg_id,
        )
        self.channel.send(ack)
    
    def _handle_ack(self, msg: LLMMessage):
        with self._lock:
            if msg.msg_id in self.pending_acks:
                del self.pending_acks[msg.msg_id]
    
    def _check_pending_acks(self):
        now = time.time()
        with self._lock:
            expired = [msg_id for msg_id, ts in self.pending_acks.items() 
                       if time.time() - ts > 10.0]
            for msg_id in expired:
                del self.pending_acks[msg_id]
                # Retry logic here
    
    def _check_pending_requests(self):
        now = time.time()
        with self._lock:
            expired = [req_id for req_id, (ts, _) in self.pending_requests.items()
                       if now - ts > 30.0]
            for req_id in expired:
                del self.pending_requests[req_id]


# ============================================================================
# LLM Interface
# ============================================================================

class LLMInterface(ABC):
    """Abstract base class for LLM/SLM backends."""
    
    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 512, 
                 temperature: float = 0.0) -> str:
        """Generate response from prompt."""
        pass
    
    @abstractmethod
    def parse_command(self, text: str) -> LLMMessage:
        """Parse text into structured LLM message."""
        pass


class LocalLLMInterface(LLMInterface):
    """Local rule-based LLM (no network required)."""
    
    def __init__(self):
        from src.control.language_interface import parse_language
        from src.control.language_to_task import text_to_task
        self.parse_language = parse_language
        self.text_to_task = text_to_task
    
    def generate(self, prompt: str, max_tokens: int = 512, 
                 temperature: float = 0.0) -> str:
        # Simple rule-based generation for SAR commands
        # In practice, this would call a local SLM
        return '{"task_id": 0, "params": {"location": "home"}}'
    
    def parse_command(self, text: str) -> LLMMessage:
        from src.control.language_interface import parse_language
        from src.control.language_to_task import text_to_task
        
        task = text_to_task(text)
        return LLMMessage(
            msg_type=MessageType.LLM_COMMAND,
            command_type=task["skill"],
            params=task["params"],
            task_id=task["task_id"],
        )


class OpenAICompatibleLLM(LLMInterface):
    """OpenAI-compatible LLM backend."""
    
    def __init__(self, base_url: str, api_key: str, model: str = "gpt-4"):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
    
    def generate(self, prompt: str, max_tokens: int = 512, 
                 temperature: float = 0.0) -> str:
        import urllib.request
        import json
        
        payload = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }).encode("utf-8")
        
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps({
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
            }).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
        )
        
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            return body["choices"][0]["message"]["content"]
    
    def parse_command(self, text: str) -> LLMMessage:
        # Use LLM to parse, fallback to local
        try:
            prompt = f"Parse SAR command: {text}\nReturn JSON with task_id, params."
            response = self.generate(f"Parse: {text}\nJSON:", max_tokens=128)
            # Parse JSON response
            import json
            data = json.loads(response)
            return LLMMessage(
                msg_type=MessageType.LLM_COMMAND,
                command_type=data.get("task_id", 0),
                params=data.get("params", {}),
            )
        except Exception:
            # Fallback
            from src.control.language_interface import parse_language
            from src.control.language_to_task import text_to_task
            task = text_to_task(text)
            return LLMMessage(
                msg_type=MessageType.LLM_COMMAND,
                command_type=task["skill"],
                params=task["params"],
            )


# ============================================================================
# LLM Controller (LLM → NN)
# ============================================================================

class LLMController:
    """LLM-side controller that manages swarm/agent via NN interface."""
    
    def __init__(self, config: CommunicationConfig, channel: CommunicationChannel,
                 llm: LLMInterface, session_id: str = ""):
        self.config = config
        self.channel = channel
        self.llm = llm
        self.session_id = str(uuid.uuid4())[:8] if not session_id else session_id
        self.episode = 0
        self.step_count = 0
        self.pending_requests: Dict[str, Tuple[float, Callable]] = {}
        self.agent_states: Dict[str, Dict] = {}  # agent_id -> state
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
    
    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._control_loop, daemon=True)
        self._thread.start()
    
    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
    
    def _control_loop(self):
        while self._running:
            # Check for NN messages
            msg = self.channel.receive(timeout=0.1, from_nn=True)
            if msg:
                self._handle_nn_message(msg)
            
            # Send periodic commands/plans
            self._send_periodic_commands()
            
            time.sleep(0.1)  # 10Hz control loop
    
    def _handle_nn_message(self, msg: NNMessage):
        if msg.msg_type == MessageType.NN_OBSERVATION:
            self._process_observation(msg)
        elif msg.msg_type == MessageType.NN_ACTION:
            self._process_action(msg)
        elif msg.msg_type == MessageType.NN_REWARD:
            self._process_reward(msg)
        elif msg.msg_type == MessageType.NN_EPISODE_END:
            self._handle_episode_end(msg)
        elif msg.msg_type == MessageType.NN_HELP_REQUEST:
            self._handle_help_request(msg)
        elif msg.msg_type == MessageType.NN_EPISODE_END:
            self._handle_episode_end(msg)
    
    def _process_observation(self, msg: NNMessage):
        # Update agent state
        agent_id = msg.session_id
        if agent_id not in self.agent_states:
            self.agent_states[agent_id] = {}
        
        self.agent_states[agent_id].update({
            "observation": msg.observation,
            "action": msg.action,
            "reward": msg.reward,
            "done": msg.done,
            "info": msg.info,
            "value": msg.value_estimate,
            "logprob": msg.action_logprob,
            "entropy": msg.entropy,
            "episode_reward": msg.episode_reward,
            "episode_length": msg.episode_length,
            "last_update": time.time(),
        })
    
    def _process_action(self, msg: NNMessage):
        pass
    
    def _process_reward(self, msg: NNMessage):
        pass
    
    def _handle_help_request(self, msg: NNMessage):
        """LLM provides help when NN requests it."""
        # Generate corrective action
        correction = LLMMessage(
            msg_type=MessageType.LLM_CORRECTION,
            session_id=msg.session_id,
            request_id=msg.msg_id,
            correction_type="action_correction",
            correction_data={
                "reason": msg.help_reason,
                "suggested_action": "hover",  # Safe default
                "confidence": 0.8,
            },
            confidence=0.8,
        )
        # Send correction
        # (would send via channel)
    
    def _handle_episode_end(self, msg: NNMessage):
        pass
    
    def _send_periodic_commands(self):
        # Send periodic goals/plans to agents
        pass
    
    def send_command(self, agent_id: str, command: str, params: Dict,
                     constraints: Dict = None) -> str:
        """Send command to specific agent."""
        msg_id = str(uuid.uuid4())[:8]
        msg = LLMMessage(
            msg_type=MessageType.LLM_COMMAND,
            session_id=self.session_id,
            request_id=msg_id,
            command_type=command,
            params=params,
            constraints=constraints or {},
        )
        # Send via channel (would need agent-specific routing)
        return msg_id
    
    def send_plan(self, agent_id: str, plan: List[Dict], plan_id: str = "") -> str:
        """Send multi-step plan to agent."""
        if not plan_id:
            plan_id = str(uuid.uuid4())[:8]
        msg = LLMMessage(
            msg_type=MessageType.LLM_PLAN,
            session_id=self.session_id,
            request_id=str(uuid.uuid4())[:8],
            plan=plan,
            plan_id=plan_id,
        )
        return plan_id
    
    def send_correction(self, agent_id: str, correction_type: str, 
                        correction_data: Dict) -> str:
        """Send correction to agent."""
        msg = LLMMessage(
            msg_type=MessageType.LLM_CORRECTION,
            session_id=self.session_id,
            request_id=str(uuid.uuid4())[:8],
            correction_type=correction_type,
            correction_data=correction_data,
        )
        return msg.msg_id
    
    def set_goal(self, agent_id: str, goal_type: str, params: Dict,
                 constraints: Dict = None) -> str:
        """Set goal for agent."""
        msg = LLMMessage(
            msg_type=MessageType.LLM_GOAL,
            session_id=self.session_id,
            request_id=str(uuid.uuid4())[:8],
            goal_type=goal_type,
            goal_params=params,
            goal_constraints=constraints or {},
        )
        return msg.msg_id
    
    def query_agent(self, agent_id: str, query_type: str, params: Dict) -> str:
        """Query agent state."""
        msg = LLMMessage(
            msg_type=MessageType.LLM_QUERY,
            session_id=self.session_id,
            request_id=str(uuid.uuid4())[:8],
            query_type=query_type,
            query_params=params,
        )
        return msg.request_id


# ============================================================================
# Bidirectional Communication Manager
# ============================================================================

class BidirectionalCommunicator:
    """Manages bidirectional NN ↔ LLM communication."""
    
    def __init__(self, config: CommunicationConfig = None):
        self.config = config or CommunicationConfig()
        self.channel = InProcessChannel(self.config)
        self.nn_interface: Optional[NeuralNetworkInterface] = None
        self.llm_controller: Optional[LLMController] = None
        self.session_id = str(uuid.uuid4())[:8]
        self._running = False
    
    def initialize_nn(self, policy, session_id: str = "") -> NeuralNetworkInterface:
        """Initialize neural network interface."""
        self.nn_interface = NeuralNetworkInterface(
            config=self.config,
            channel=self.channel,
            policy=policy,
            session_id=session_id or str(uuid.uuid4())[:8],
        )
        return self.nn_interface
    
    def initialize_llm(self, llm: LLMInterface = None, session_id: str = "") -> LLMController:
        """Initialize LLM controller."""
        if llm is None:
            llm = LocalLLMInterface()
        self.llm_controller = LLMController(
            config=self.config,
            channel=self.channel,
            llm=llm,
            session_id=session_id or str(uuid.uuid4())[:8],
        )
        return self.llm_controller
    
    def start(self):
        """Start bidirectional communication."""
        if self.nn_interface:
            self.nn_interface.start()
        if self.llm_controller:
            self.llm_controller.start()
    
    def stop(self):
        if self.nn_interface:
            self.nn_interface.stop()
        if self.llm_controller:
            self.llm_controller.stop()
    
    def send_nn_to_llm(self, msg: NNMessage) -> bool:
        """Send message from NN to LLM."""
        return self.channel.send(msg)
    
    def send_llm_to_nn(self, msg: LLMMessage) -> bool:
        """Send message from LLM to NN."""
        return self.channel.send(msg)
    
    def get_nn_message(self, timeout: float = 1.0) -> Optional[NNMessage]:
        """Get message from NN."""
        return self.channel.receive(timeout=timeout, from_nn=True)
    
    def get_llm_message(self, timeout: float = 1.0) -> Optional[LLMMessage]:
        """Get message from LLM."""
        return self.channel.receive(timeout=timeout, from_nn=False)


# ============================================================================
# Integration with Existing SAR System
# ============================================================================

class SARCommunicationManager:
    """High-level manager for SAR drone communication."""
    
    def __init__(self, config: CommunicationConfig = None):
        self.config = config or CommunicationConfig()
        self.communicator = BidirectionalCommunicator(config)
        self.session_id = str(uuid.uuid4())[:8]
    
    def setup_sar_system(self, policy, llm=None):
        """Setup complete SAR communication system."""
        self.communicator.initialize_nn(policy=policy, session_id=self.session_id)
        self.communicator.initialize_llm(llm=llm, session_id=self.session_id)
        self.communicator.start()
    
    def send_sar_command(self, command: str, params: Dict = None) -> str:
        """Send SAR command via LLM."""
        from src.control.language_to_task import text_to_task
        task = text_to_task(command)
        msg = LLMMessage(
            msg_type=MessageType.LLM_COMMAND,
            command_type=task["skill"],
            params=task["params"],
        )
        return self.communicator.send_llm_to_nn(msg)
    
    def send_language_command(self, text: str) -> str:
        """Send natural language command."""
        return self.send_sar_command(text)
    
    def get_nn_observation(self, timeout: float = 1.0) -> Optional[NNMessage]:
        """Get observation from neural network."""
        return self.communicator.get_nn_message(timeout=timeout)
    
    def get_llm_response(self, timeout: float = 1.0) -> Optional[LLMMessage]:
        """Get response from LLM."""
        return self.communicator.get_llm_message(timeout=timeout)
    
    def shutdown(self):
        """Shutdown communication."""
        # Implementation would stop threads, close channels
        pass


# ============================================================================
# Example Usage & Testing
# ============================================================================

def create_sar_communication_system(policy, llm=None) -> SARCommunicationManager:
    """Factory function to create complete SAR communication system."""
    config = CommunicationConfig(
        llm_timeout_s=10.0,
        nn_timeout_s=5.0,
        heartbeat_interval_s=1.0,
        validate_sar_only=True,
    )
    manager = SARCommunicationManager(config)
    manager.setup_sar_system(policy, llm)
    return manager


def demo_bidirectional_communication():
    """Demonstrate bidirectional NN ↔ LLM communication."""
    print("=== DroneNav-SAR Bidirectional Communication Demo ===\n")
    
    # Create mock policy
    from src.rl.policies import ActorCritic
    policy = ActorCritic()
    
    # Create communication system
    manager = create_sar_communication_system(policy)
    manager.setup_sar_system(policy)
    
    # Test 1: Send language command
    print("Test 1: Send language command")
    msg_id = manager.send_language_command("deliver medkit to room 2")
    print(f"  Sent command, msg_id: {msg_id}")
    
    # Test 2: Get NN observation
    print("\nTest 2: Get NN observation")
    obs_msg = manager.get_nn_observation(timeout=1.0)
    if obs_msg:
        print(f"  Received observation: shape={obs_msg.observation.shape if obs_msg.observation is not None else None}")
    
    # Test 3: Get LLM response
    print("\nTest 3: Get LLM response")
    llm_msg = manager.get_llm_response(timeout=1.0)
    if llm_msg:
        print(f"  Received LLM message: {llm_msg.msg_type}")
    
    print("\n✅ Bidirectional communication demo complete")


if __name__ == "__main__":
    demo_bidirectional_communication()